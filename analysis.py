import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List


COLORS = {'BLUE': '#3D6FFF',
          'RED': '#FF3D3D',
          'ORANGE': '#FF8E35',
          'PURPLE': '#BB58FF',
          'GREEN': '#32CD32',
          'YELLOW': '#F9DB00',
          'PINK': '#FFC0CB',
          'BROWN': '#8B4513',
          'CYAN': '#00FFFF',
}

def adjust_color(color: str, 
                 factor: float = 0.5) -> str:
    """
    Adjust the brightness of a color by a specified factor.

    Parameters:
    -----------
    color (str):
        The color to adjust.
    
    factor (float, optional):
        The factor by which to adjust the color. Defaults to 0.5.
        If negative, the color will be lightened.

    
    Returns:
    --------
    str:
        The adjusted color.
    
    Examples:
    ---------
    >>> adjust_color(COLORS['BLUE'], 0.5)
    >>> adjust_color(COLORS['BLUE'], -0.5)
    """

    assert(factor >= -1 and factor <= 1), "Factor must be between -1 and 1."

    r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    if factor >= 0:
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
    else:
        factor = abs(factor)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)

    return '#%02x%02x%02x' % (r, g, b)

def create_gradient(color: str, 
                    n: int) -> List[str]:
    """
    Create a gradient of colors from a base color. Color given is the middle color. 
    Lighter colors are before the middle color and darker colors are after the middle color.

    Parameters:
    -----------

    color (str):
        The base color from which to create the gradient.

    n (int):
        The number of colors in the gradient.

    Returns:
    --------
    List[str]:
        A list of colors representing the gradient.

    Examples:
    ---------
    >>> create_gradient(COLORS['BLUE'], 5)
    """

    gradient = [color]
    for i in range(1, n//2 + 1):
        gradient.insert(0, adjust_color(color, -i/n))
        gradient.append(adjust_color(color, (n-i)/n))

    
    return gradient


def save_plot(filepath: str, 
              **kwargs) -> None:
    """
    Save the current plot to a file.

    Parameters:
    -----------
    filepath (str):
        The file path to save the plot.

    **kwargs:
        Additional keyword arguments for saving the plot.

    Returns:
    --------
    None
    """
    if not filepath.endswith('.png'):
        filepath += '.png'
    plt.savefig(filepath, bbox_inches="tight", **kwargs)

df = pd.read_json('data_spotify/StreamingHistory_music_0.json')

df_msPlayed = df.drop(columns='endTime')

def plot_hist_top_n(df: pd.DataFrame,
                    column: str = 'ARTIST', 
                    n: int = 10,
                    filepath: str = None,
                    **kwargs) -> None:
    
    column = 'artistName' if 'ARTIST' in column.upper() else 'trackName'
    name = 'Artists' if column == 'artistName' else 'Tracks'

    if column == 'artistName':
        df = df.groupby('artistName').agg({'msPlayed': 'sum'}).reset_index()
    else:
        df = df.groupby(['artistName', 'trackName']).agg({'msPlayed': 'sum'}).reset_index()

    df = df.sort_values('msPlayed', ascending=False)

    df = df.head(n).iloc[::-1]
    print(df)

    # Create a gradient of colors
    color = kwargs.get('color', COLORS['BLUE'])
    gradient = create_gradient(color, n)

    fig, ax = plt.subplots()

    # Plot the top n artists
    for i in range(n):
        title = df[column].iloc[i] if len(df[column].iloc[i]) < 20 else df[column].iloc[i][:20] + '...'
        msPlayed = df['msPlayed'].iloc[i]
        ax.barh(title, msPlayed, color=gradient[i])

    ax.set_xlabel('Total ms Played')
    ax.set_ylabel(name)
    ax.set_title(f'Top {n} {name} by Total ms Played')

    if filepath:
        save_plot(filepath)

    plt.show()

n = 20
plot_hist_top_n(df_msPlayed,
                'artist', 
                n, 
                f'top_{n}_artists', 
                color= COLORS['BLUE'])
