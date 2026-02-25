import torch
from sklearn.feature_extraction.text import TfidfTransformer


class TFIDFSelector(object):
    def __init__(
        self,
        interactions: torch.Tensor,
        max_hist: int,
    ):
        super().__init__()
        self.interactions = interactions
        self.max_hist = max_hist

    def __call__(self, **kwargs):
        # padding idx
        num_anchor, num_target = self.interactions.shape

        # compute tfidf
        tfidf = TfidfTransformer(norm=None)
        tfidf_matrix = tfidf.fit_transform(self.interactions)

        # ndarray -> tensor
        kwargs = dict(
            data=tfidf_matrix.toarray(),
            dtype=torch.float32,
        )
        tfidf_matrix_dense = torch.tensor(**kwargs)

        # select top-k indices
        topk_indices = []
        for row in range(len(tfidf_matrix_dense)):
            hist_count = int(self.interactions[row].sum().item())
            
            # padding only
            if hist_count == 0:
                indices = torch.tensor([num_target], dtype=torch.long)
                topk_indices.append(indices.to(torch.long))
            
            # all
            elif hist_count <= self.max_hist:
                indices = self.interactions[row].nonzero(as_tuple=True)[0]
                topk_indices.append(indices.to(torch.long))
            
            # top-k selection
            else:
                vals, indices = torch.topk(tfidf_matrix_dense[row], k=self.max_hist)
                topk_indices.append(indices.to(torch.long))

        return topk_indices