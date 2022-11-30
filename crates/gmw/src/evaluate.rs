pub mod and {
    use crate::mul_triple::MulTriple;

    #[inline]
    pub fn compute_shares(x: bool, y: bool, mt: &MulTriple) -> (bool, bool) {
        let d = x ^ mt.a();
        let e = y ^ mt.b();
        (d, e)
    }

    #[inline]
    pub fn evaluate(d: [bool; 2], e: [bool; 2], mt: MulTriple, party_id: usize) -> bool {
        let d = d[0] ^ d[1];
        let e = e[0] ^ e[1];
        let res = d & mt.b() ^ e & mt.a() ^ mt.c();
        if party_id == 0 {
            res ^ d & e
        } else {
            res
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::evaluate::and;
    use crate::mul_triple::MulTriple;

    #[test]
    fn and_eval() {
        let p0 = [false, true];
        let p1 = [true, false];
        let mt0 = MulTriple::zeros();
        let mt1 = MulTriple::zeros();
        let shares_0 = and::compute_shares(p0[0], p0[1], &mt0);
        let shares_1 = and::compute_shares(p1[0], p1[1], &mt1);
        let a0 = and::evaluate([shares_0.0, shares_1.0], [shares_0.1, shares_1.1], mt0, 0);
        let a1 = and::evaluate([shares_0.0, shares_1.0], [shares_0.1, shares_1.1], mt1, 1);
        assert_eq!(true, a0 ^ a1);
    }
}
