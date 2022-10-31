import pandas as pd

class MyDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def append_rows(self, rows):
        dfaux = pd.DataFrame(rows, columns=self.columns)
        return self.append(dfaux, ignore_index=True)

    def concat(self, dfs):
        return pd.concat(dfs)


