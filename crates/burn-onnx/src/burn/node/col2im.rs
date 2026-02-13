use super::prelude::*;

impl NodeCodegen for onnx_ir::col2im::Col2ImNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        let image_shape = &self.config.image_shape;
        let block_shape = &self.config.block_shape;
        let strides = &self.config.strides;
        let dilations = &self.config.dilations;
        let pads = &self.config.pads;

        let num_spatial_dims = image_shape.len();

        // Split pads into begin/end
        let pads_begin: Vec<usize> = pads[..num_spatial_dims].to_vec();
        let pads_end: Vec<usize> = pads[num_spatial_dims..].to_vec();

        // Compute effective block sizes: d * (k - 1) + 1
        let effective_blocks: Vec<usize> = block_shape
            .iter()
            .zip(dilations.iter())
            .map(|(&b, &d)| d * (b - 1) + 1)
            .collect();

        // Compute output windows count per dimension
        // L_i = (img + pad_begin + pad_end - effective) / stride + 1
        let output_counts: Vec<usize> = (0..num_spatial_dims)
            .map(|i| {
                (image_shape[i] + pads_begin[i] + pads_end[i] - effective_blocks[i]) / strides[i]
                    + 1
            })
            .collect();

        let total_windows: usize = output_counts.iter().product();
        let block_product: usize = block_shape.iter().product();
        let total_input_elements = block_product * total_windows;

        // Compute Padded Output Shape dimensions
        let padded_dims: Vec<usize> = (0..num_spatial_dims)
            .map(|i| image_shape[i] + pads_begin[i] + pads_end[i])
            .collect();

        // Calculate the linear indices for scatter-add
        // We compute where each element of the input (flattened block * windows) goes in the flattened padded output.
        // Input layout: [Batch, Channel, BlockElements, Windows] -> Flattened last 2 dims: [BlockElements, Windows]
        // But Burn reshape/flatten is usually C-order (row-major).
        // Input tensor is [N, C, BlockProd, L] -> reshape to [N, C, BlockProd * L]
        // So we iterate: for window in 0..windows { for block_elem in 0..block_product { ... } } (?)
        // Wait, reshape [N, C, Block, L] -> [N, C, Block*L] means inner dimension is L.
        // So iterate block_elem outer, window inner? No, Burn/Numpy default layout is standard (last dim contiguous).
        // So [d0, d1, d2, d3] -> [d0, d1, d2*d3] means d3 is the fastest changing index.
        // So index = block_idx * total_windows + window_idx

        // BUT, Col2Im input is usually [N, C*BlockProd, L] in ONNX spec.
        // My onnx-ir says: "Input data tensor from Im2Col, shape [N, C * product(block_shape), L]"
        // So we interpret it as [N, C, BlockProd, L] for reshaping purposes (logic in type inference confirms this).
        // So element at index `i` in flattened spatial dim corresponds to:
        //   block_idx = i / total_windows
        //   window_idx = i % total_windows

        let mut scatter_indices = vec![0i64; total_input_elements];

        for (i, index) in scatter_indices.iter_mut().enumerate() {
            let block_idx = i / total_windows;
            let window_idx = i % total_windows;

            // Reconstruct spatial positions from window_idx and block_idx
            // General N-D logic
            let mut w_rem = window_idx;
            let mut b_rem = block_idx;

            let mut flat_output_index = 0;
            let mut stride_accumulator = 1;

            // Iterate dimensions backwards (W then H) for standard layout
            for dim in (0..num_spatial_dims).rev() {
                // Window position in this dimension
                let w_pos = w_rem % output_counts[dim];
                w_rem /= output_counts[dim];

                // Block offset in this dimension
                let b_pos = b_rem % block_shape[dim];
                b_rem /= block_shape[dim];

                // Calculate position in Padded Output
                // pos = w_pos * stride + b_pos * dilation
                let pad_pos = w_pos * strides[dim] + b_pos * dilations[dim];

                // Add to flat index
                flat_output_index += pad_pos * stride_accumulator;
                stride_accumulator *= padded_dims[dim];
            }

            *index = flat_output_index as i64;
        }

        // Create constant tensor from logic
        let _indices_len = scatter_indices.len();
        let indices_tokens = quote! {
            TensorData::from(&[#(#scatter_indices),*] as &[i64])
        };

        let padded_size: usize = padded_dims.iter().product();

        // 6. Slice instructions to crop padding
        // canvas.slice([0..N, 0..C, pad_h..pad_h+H, pad_w..pad_w+W])
        // Since we flatten spatial dims for scatter, we need to reshape back to spatial before slicing.

        // Padded shape for reshape
        let padded_shape_tokens = match num_spatial_dims {
            1 => quote! { [batch_size, channels, #padded_size] },
            2 => {
                let h_pad = padded_dims[0];
                let w_pad = padded_dims[1];
                quote! { [batch_size, channels, #h_pad, #w_pad] }
            }
            _ => panic!("Unsupported dimensions"),
        };

        // Crop slices
        let slice_ranges = match num_spatial_dims {
            1 => {
                let p_begin = pads_begin[0];
                let shape = image_shape[0];
                let end = p_begin + shape;
                quote! { [0..batch_size, 0..channels, #p_begin..#end] }
            }
            2 => {
                let h_begin = pads_begin[0];
                let h_shape = image_shape[0];
                let h_end = h_begin + h_shape;

                let w_begin = pads_begin[1];
                let w_shape = image_shape[1];
                let w_end = w_begin + w_shape;

                quote! { [0..batch_size, 0..channels, #h_begin..#h_end, #w_begin..#w_end] }
            }
            _ => panic!("Unsupported dimensions"),
        };

        // Output image shape for result (validation/verification)
        // let output_shape = ...

        quote! {
            let #output = {
                let [batch_size, col_channels, _l] = #input.shape().dims();
                let channels = col_channels / #block_product;
                let device = #input.device();

                // 1. Convert input to flattened [N, C, BlockProd * Windows]
                // Note: col2im input is [N, C*Block, L], reshape to [N, C, Block*L] works if contiguous
                let input_flat = #input.reshape([batch_size, channels, #total_input_elements]);

                // 2. Create output canvas (Padded, Flattened)
                // Shape: [N, C, PaddedTotal]
                let mut canvas = Tensor::<B, 3>::zeros([batch_size, channels, #padded_size], &device);

                // 3. Create Indices Tensor [BlockProd * Windows]
                let indices = Tensor::<B, 1, Int>::from_data(#indices_tokens, &device);

                // 4. Expand indices to [N, C, BlockProd * Windows]
                let indices_expanded = indices
                    .reshape([1, 1, -1])
                    .expand([batch_size, channels, #total_input_elements]);

                // 5. Scatter Add (dim 2)
                let canvas = canvas.scatter(2, indices_expanded, input_flat, burn::tensor::IndexingUpdateOp::Add);

                // 6. Reshape to Padded Spatial and Crop
                let canvas = canvas.reshape(#padded_shape_tokens);

                canvas.slice(#slice_ranges)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::col2im::{Col2ImConfig, Col2ImNodeBuilder};

    #[test]
    fn test_col2im_2d_basic() {
        let config = Col2ImConfig::new(
            vec![5, 5],       // image_shape
            vec![2, 2],       // block_shape
            vec![1, 1],       // dilations
            vec![0, 0, 0, 0], // pads
            vec![1, 1],       // strides
        );
        let node = Col2ImNodeBuilder::new("col2im1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let [batch_size, col_channels, _l] = input.shape().dims();
                let channels = col_channels / 4usize;
                let device = input.device();
                let input_flat = input.reshape([batch_size, channels, 64usize]);
                let mut canvas = Tensor::<B, 3>::zeros([batch_size, channels, 25usize], &device);
                let indices = Tensor::<
                    B,
                    1,
                    Int,
                >::from_data(
                    TensorData::from(
                        &[
                            0i64,
                            1i64,
                            2i64,
                            3i64,
                            5i64,
                            6i64,
                            7i64,
                            8i64,
                            10i64,
                            11i64,
                            12i64,
                            13i64,
                            15i64,
                            16i64,
                            17i64,
                            18i64,
                            1i64,
                            2i64,
                            3i64,
                            4i64,
                            6i64,
                            7i64,
                            8i64,
                            9i64,
                            11i64,
                            12i64,
                            13i64,
                            14i64,
                            16i64,
                            17i64,
                            18i64,
                            19i64,
                            5i64,
                            6i64,
                            7i64,
                            8i64,
                            10i64,
                            11i64,
                            12i64,
                            13i64,
                            15i64,
                            16i64,
                            17i64,
                            18i64,
                            20i64,
                            21i64,
                            22i64,
                            23i64,
                            6i64,
                            7i64,
                            8i64,
                            9i64,
                            11i64,
                            12i64,
                            13i64,
                            14i64,
                            16i64,
                            17i64,
                            18i64,
                            19i64,
                            21i64,
                            22i64,
                            23i64,
                            24i64,
                        ] as &[i64],
                    ),
                    &device,
                );
                let indices_expanded = indices
                    .reshape([1, 1, -1])
                    .expand([batch_size, channels, 64usize]);
                let canvas = canvas
                    .scatter(
                        2,
                        indices_expanded,
                        input_flat,
                        burn::tensor::IndexingUpdateOp::Add,
                    );
                let canvas = canvas.reshape([batch_size, channels, 5usize, 5usize]);
                canvas.slice([0..batch_size, 0..channels, 0usize..5usize, 0usize..5usize])
            };
            output
        }
        "###);
    }

    #[test]
    fn test_col2im_2d_with_padding() {
        let config = Col2ImConfig::new(
            vec![5, 5],       // image_shape
            vec![2, 2],       // block_shape
            vec![1, 1],       // dilations
            vec![1, 1, 1, 1], // pads [t, l, b, r]
            vec![1, 1],       // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_pad")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let [batch_size, col_channels, _l] = input.shape().dims();
                let channels = col_channels / 4usize;
                let device = input.device();
                let input_flat = input.reshape([batch_size, channels, 144usize]);
                let mut canvas = Tensor::<B, 3>::zeros([batch_size, channels, 49usize], &device);
                let indices = Tensor::<
                    B,
                    1,
                    Int,
                >::from_data(
                    TensorData::from(
                        &[
                            0i64,
                            1i64,
                            2i64,
                            3i64,
                            4i64,
                            5i64,
                            7i64,
                            8i64,
                            9i64,
                            10i64,
                            11i64,
                            12i64,
                            14i64,
                            15i64,
                            16i64,
                            17i64,
                            18i64,
                            19i64,
                            21i64,
                            22i64,
                            23i64,
                            24i64,
                            25i64,
                            26i64,
                            28i64,
                            29i64,
                            30i64,
                            31i64,
                            32i64,
                            33i64,
                            35i64,
                            36i64,
                            37i64,
                            38i64,
                            39i64,
                            40i64,
                            1i64,
                            2i64,
                            3i64,
                            4i64,
                            5i64,
                            6i64,
                            8i64,
                            9i64,
                            10i64,
                            11i64,
                            12i64,
                            13i64,
                            15i64,
                            16i64,
                            17i64,
                            18i64,
                            19i64,
                            20i64,
                            22i64,
                            23i64,
                            24i64,
                            25i64,
                            26i64,
                            27i64,
                            29i64,
                            30i64,
                            31i64,
                            32i64,
                            33i64,
                            34i64,
                            36i64,
                            37i64,
                            38i64,
                            39i64,
                            40i64,
                            41i64,
                            7i64,
                            8i64,
                            9i64,
                            10i64,
                            11i64,
                            12i64,
                            14i64,
                            15i64,
                            16i64,
                            17i64,
                            18i64,
                            19i64,
                            21i64,
                            22i64,
                            23i64,
                            24i64,
                            25i64,
                            26i64,
                            28i64,
                            29i64,
                            30i64,
                            31i64,
                            32i64,
                            33i64,
                            35i64,
                            36i64,
                            37i64,
                            38i64,
                            39i64,
                            40i64,
                            42i64,
                            43i64,
                            44i64,
                            45i64,
                            46i64,
                            47i64,
                            8i64,
                            9i64,
                            10i64,
                            11i64,
                            12i64,
                            13i64,
                            15i64,
                            16i64,
                            17i64,
                            18i64,
                            19i64,
                            20i64,
                            22i64,
                            23i64,
                            24i64,
                            25i64,
                            26i64,
                            27i64,
                            29i64,
                            30i64,
                            31i64,
                            32i64,
                            33i64,
                            34i64,
                            36i64,
                            37i64,
                            38i64,
                            39i64,
                            40i64,
                            41i64,
                            43i64,
                            44i64,
                            45i64,
                            46i64,
                            47i64,
                            48i64,
                        ] as &[i64],
                    ),
                    &device,
                );
                let indices_expanded = indices
                    .reshape([1, 1, -1])
                    .expand([batch_size, channels, 144usize]);
                let canvas = canvas
                    .scatter(
                        2,
                        indices_expanded,
                        input_flat,
                        burn::tensor::IndexingUpdateOp::Add,
                    );
                let canvas = canvas.reshape([batch_size, channels, 7usize, 7usize]);
                canvas.slice([0..batch_size, 0..channels, 1usize..6usize, 1usize..6usize])
            };
            output
        }
        "###);
    }

    #[test]
    fn test_col2im_2d_with_strides() {
        let config = Col2ImConfig::new(
            vec![6, 6],       // image_shape
            vec![2, 2],       // block_shape
            vec![1, 1],       // dilations
            vec![0, 0, 0, 0], // pads
            vec![2, 2],       // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_stride")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let [batch_size, col_channels, _l] = input.shape().dims();
                let channels = col_channels / 4usize;
                let device = input.device();
                let input_flat = input.reshape([batch_size, channels, 36usize]);
                let mut canvas = Tensor::<B, 3>::zeros([batch_size, channels, 36usize], &device);
                let indices = Tensor::<
                    B,
                    1,
                    Int,
                >::from_data(
                    TensorData::from(
                        &[
                            0i64,
                            2i64,
                            4i64,
                            12i64,
                            14i64,
                            16i64,
                            24i64,
                            26i64,
                            28i64,
                            1i64,
                            3i64,
                            5i64,
                            13i64,
                            15i64,
                            17i64,
                            25i64,
                            27i64,
                            29i64,
                            6i64,
                            8i64,
                            10i64,
                            18i64,
                            20i64,
                            22i64,
                            30i64,
                            32i64,
                            34i64,
                            7i64,
                            9i64,
                            11i64,
                            19i64,
                            21i64,
                            23i64,
                            31i64,
                            33i64,
                            35i64,
                        ] as &[i64],
                    ),
                    &device,
                );
                let indices_expanded = indices
                    .reshape([1, 1, -1])
                    .expand([batch_size, channels, 36usize]);
                let canvas = canvas
                    .scatter(
                        2,
                        indices_expanded,
                        input_flat,
                        burn::tensor::IndexingUpdateOp::Add,
                    );
                let canvas = canvas.reshape([batch_size, channels, 6usize, 6usize]);
                canvas.slice([0..batch_size, 0..channels, 0usize..6usize, 0usize..6usize])
            };
            output
        }
        "###);
    }

    #[test]
    fn test_col2im_2d_with_dilation() {
        let config = Col2ImConfig::new(
            vec![5, 5],       // image_shape
            vec![2, 2],       // block_shape
            vec![2, 2],       // dilations
            vec![0, 0, 0, 0], // pads
            vec![1, 1],       // strides
        );
        let node = Col2ImNodeBuilder::new("col2im_dil")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
            let output = {
                let [batch_size, col_channels, _l] = input.shape().dims();
                let channels = col_channels / 4usize;
                let device = input.device();
                let input_flat = input.reshape([batch_size, channels, 36usize]);
                let mut canvas = Tensor::<B, 3>::zeros([batch_size, channels, 25usize], &device);
                let indices = Tensor::<
                    B,
                    1,
                    Int,
                >::from_data(
                    TensorData::from(
                        &[
                            0i64,
                            1i64,
                            2i64,
                            5i64,
                            6i64,
                            7i64,
                            10i64,
                            11i64,
                            12i64,
                            2i64,
                            3i64,
                            4i64,
                            7i64,
                            8i64,
                            9i64,
                            12i64,
                            13i64,
                            14i64,
                            10i64,
                            11i64,
                            12i64,
                            15i64,
                            16i64,
                            17i64,
                            20i64,
                            21i64,
                            22i64,
                            12i64,
                            13i64,
                            14i64,
                            17i64,
                            18i64,
                            19i64,
                            22i64,
                            23i64,
                            24i64,
                        ] as &[i64],
                    ),
                    &device,
                );
                let indices_expanded = indices
                    .reshape([1, 1, -1])
                    .expand([batch_size, channels, 36usize]);
                let canvas = canvas
                    .scatter(
                        2,
                        indices_expanded,
                        input_flat,
                        burn::tensor::IndexingUpdateOp::Add,
                    );
                let canvas = canvas.reshape([batch_size, channels, 5usize, 5usize]);
                canvas.slice([0..batch_size, 0..channels, 0usize..5usize, 0usize..5usize])
            };
            output
        }
        "###);
    }

    #[test]
    fn test_col2im_1d_basic() {
        let config = Col2ImConfig::new(
            vec![10],   // image_shape
            vec![3],    // block_shape
            vec![1],    // dilations
            vec![0, 0], // pads
            vec![1],    // strides
        );
        let node = Col2ImNodeBuilder::new("col2im1d")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = {
                let [batch_size, col_channels, _l] = input.shape().dims();
                let channels = col_channels / 3usize;
                let device = input.device();
                let input_flat = input.reshape([batch_size, channels, 24usize]);
                let mut canvas = Tensor::<B, 3>::zeros([batch_size, channels, 10usize], &device);
                let indices = Tensor::<
                    B,
                    1,
                    Int,
                >::from_data(
                    TensorData::from(
                        &[
                            0i64,
                            1i64,
                            2i64,
                            3i64,
                            4i64,
                            5i64,
                            6i64,
                            7i64,
                            1i64,
                            2i64,
                            3i64,
                            4i64,
                            5i64,
                            6i64,
                            7i64,
                            8i64,
                            2i64,
                            3i64,
                            4i64,
                            5i64,
                            6i64,
                            7i64,
                            8i64,
                            9i64,
                        ] as &[i64],
                    ),
                    &device,
                );
                let indices_expanded = indices
                    .reshape([1, 1, -1])
                    .expand([batch_size, channels, 24usize]);
                let canvas = canvas
                    .scatter(
                        2,
                        indices_expanded,
                        input_flat,
                        burn::tensor::IndexingUpdateOp::Add,
                    );
                let canvas = canvas.reshape([batch_size, channels, 10usize]);
                canvas.slice([0..batch_size, 0..channels, 0usize..10usize])
            };
            output
        }
        "###);
    }
}
