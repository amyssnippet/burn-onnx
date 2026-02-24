use super::prelude::*;

pub(crate) fn align_binary_operands_for_broadcast(
    lhs_expr: TokenStream,
    lhs_rank: usize,
    rhs_expr: TokenStream,
    rhs_rank: usize,
) -> (TokenStream, TokenStream) {
    if lhs_rank == rhs_rank {
        return (lhs_expr, rhs_expr);
    }

    if lhs_rank > rhs_rank {
        let num_dims = lhs_rank - rhs_rank;
        let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
        (lhs_expr, quote! { #rhs_expr.unsqueeze_dims(&[#(#dims),*]) })
    } else {
        let num_dims = rhs_rank - lhs_rank;
        let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
        (quote! { #lhs_expr.unsqueeze_dims(&[#(#dims),*]) }, rhs_expr)
    }
}

pub(crate) fn align_rhs_for_lhs_rank(
    rhs_expr: TokenStream,
    lhs_rank: usize,
    rhs_rank: usize,
    axis: Option<i64>,
) -> TokenStream {
    if lhs_rank <= rhs_rank {
        return rhs_expr;
    }

    if rhs_rank == 1 && lhs_rank > 1 {
        let axis = axis.unwrap_or(1);
        let axis_norm = if axis < 0 {
            (lhs_rank as i64 + axis) as usize
        } else {
            axis as usize
        };

        let dims: Vec<isize> = (0..lhs_rank)
            .filter(|&i| i != axis_norm)
            .map(|i| i as isize)
            .collect();
        quote! { (#rhs_expr).unsqueeze_dims(&[#(#dims),*]) }
    } else {
        let num_dims = lhs_rank - rhs_rank;
        let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
        quote! { (#rhs_expr).unsqueeze_dims(&[#(#dims),*]) }
    }
}
