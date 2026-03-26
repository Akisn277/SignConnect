feat_min  = flat_tr.min(axis=0)
    feat_max  = flat_tr.max(axis=0)
    feat_rng  = np.where((feat_max - feat_min) == 0, 1.0, feat_max - feat_min)

    def norm(arr):
        N2, T2, F2 = arr.shape
        return ((arr.reshape(-1, F2) - feat_min) / feat_rng).reshape(N2, T2, F2)