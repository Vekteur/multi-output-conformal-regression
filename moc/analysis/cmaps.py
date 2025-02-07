import seaborn as sns


def get_cmap(df, cmap):
    names = df.name.unique()
    if cmap is None:
        return {name: color for name, color in zip(names, sns.color_palette(n_colors=len(names)))}
    elif type(cmap) == str:
        cmap = {
            
        }[cmap](df)
    return cmap
