import os

import imageio.v2 as imageio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from functools import partial

from typing import List, Tuple, Union


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

def get_scaled_size(size: int,
                    n: int) -> int:
    """
    Get the scaled size of a text.

    Parameters:
    -----------
    size (int):
        The original size of the text.

    n (int):
        The number of elements to scale the size.

    Returns:
    --------
    int:
        The scaled size of the text.
    """
    n = 1 if n <= 0 else n
    return size / n if size / n < 15 else 15



df = pd.read_json('data_spotify/StreamingHistory_music_0.json')

fig, ax = plt.subplots(figsize=(15, 8))

def get_most_listened(date: str,
                      df_input: pd.DataFrame,
                      column: str = 'ARTIST',
                      previous_day: bool = True) -> str:
    """
    Get the most listened artist or track on a given date.
    
    Parameters:
    -----------
    date (str):
        The date for which to get the most listened artist or track.
    
    df_input (pd.DataFrame):
        The dataframe containing the listening history.
        
    column (str, optional):
        The column to use for the most listened artist or track. 
        Must be either 'ARTIST' or 'TRACK'. Defaults to 'ARTIST'.
        
    previous_day (bool, optional):
        Whether to get the most listened artist or track on the previous days. Defaults to True.
        
    Returns:
    --------
    str:
        The most listened artist or track on the given date.
        
    Examples:
    ---------
    >>> get_most_listened('2021-01-01', df, 'ARTIST', True)
    >>> get_most_listened('2021-01-01', df, 'TRACK', False)
    """

    df = df_input.copy()

    df['endTime'] = pd.to_datetime(df['endTime']).dt.date

    date = pd.to_datetime(date).date()
    df = df[df['endTime'] <= date] if previous_day else df[df['endTime'] == date]
    date = date.strftime('%Y-%m-%d')

    if column == 'ARTIST':
        most_listened = df['artistName'].value_counts().idxmax()
    elif column == 'TRACK':
        most_listened = df['trackName'].value_counts().idxmax()
    else:
        raise ValueError("Column must be either 'ARTIST' or 'TRACK'.")
    
    return most_listened

                      

def plot_hist_top_n(date: str,
                    df_input: pd.DataFrame,
                    column: str = 'ARTIST', 
                    n: int = 10,
                    unit: str = 'ms',
                    previous_day: bool = True,
                    filepath: str = None,
                    **kwargs) -> None:
    
    df = df_input.copy()

    df['endTime'] = pd.to_datetime(df['endTime']).dt.date

    date = pd.to_datetime(date).date()
    df = df[df['endTime'] <= date] if previous_day else df[df['endTime'] == date]
    date = date.strftime('%Y-%m-%d')
    
    df['msPlayed'] = df['msPlayed'].astype(float)
    
    if unit == 's':
        df['msPlayed'] = df['msPlayed'] / 1000
        type_name = 'Seconds'
    elif unit == 'm':
        df['msPlayed'] = df['msPlayed'] / 1000 / 60
        type_name = 'Minutes'
    elif unit == 'h':
        df['msPlayed'] = df['msPlayed'] / 1000 / 60 / 60
        type_name = 'Hours'
    else:   
        type_name = 'Milliseconds'
    
    column, other_column, logo_dx = ('artistName', '', 0.62)  if 'ARTIST' in column.upper() else ('trackName', 'artistName', 0.615)
    name = 'Artists' if column == 'artistName' else 'Tracks'

    if column == 'artistName':
        df = df.groupby('artistName').agg({'msPlayed': 'sum'}).reset_index()
    else:
        df = df.groupby(['artistName', 'trackName']).agg({'msPlayed': 'sum'}).reset_index()

    df = df.sort_values('msPlayed', ascending=False)

    df = df.head(n).iloc[::-1]

    n = min(n, len(df))

    # Create a gradient of colors
    color = kwargs.get('color', COLORS['BLUE'])
    gradient = create_gradient(color, n)

    ax.clear()

    dx = df['msPlayed'].max() / 200

    # Plot the top n artists
    for i in range(n):
        msPlayed = df['msPlayed'].iloc[i]
        principal = df[column].iloc[i] if len(df[column].iloc[i]) < 20 else df[column].iloc[i][:20] + '...'
        if other_column != '':
            secondary = df[other_column].iloc[i] if len(df[other_column].iloc[i]) < 20 else df[other_column].iloc[i][:20] + '...'
        ax.barh(principal, msPlayed, color=gradient[i])
        ax.text(msPlayed-dx, i, principal, size=get_scaled_size(120, n-3), 
                weight=600, ha='right', va='bottom')
        if other_column != '':
            ax.text(msPlayed-dx, i-.25, secondary, size=get_scaled_size(100, n-3), 
                    color='#444444', ha='right', va='baseline')
        ax.text(msPlayed+dx, i,	 f'{msPlayed:,.0f}', size=get_scaled_size(120, n-3), 
                ha='left', va='center')

    ax.text(1, 0.05, date, transform=ax.transAxes, 
			color=gradient[-1], size=46, ha='right',
			weight=800)
    ax.text(0, 1.05, f'Time played ({type_name.lower()})',
			transform=ax.transAxes, size=12,
			color=gradient[-1])
    
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.08, f'Most-listened-to {name.lower()} from Spotify',
            transform=ax.transAxes, size=24, weight=600, ha='left')
    
    spotify_logo = plt.imread('spotify.png')
    imagebox = OffsetImage(spotify_logo, zoom=0.05)
    ab = AnnotationBbox(imagebox, (logo_dx, 1.1), xycoords='axes fraction', frameon=False)
    ax.add_artist(ab)
    
    ax.text(1, 0, 'by @ydepledt', 
            transform=ax.transAxes, ha='right', color='#777777', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)

    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    if filepath:
        save_plot(filepath)

def create_gif_top_n_evolution(df_input: pd.DataFrame,
                               column: str,
                               time_range: Union[str, Tuple[str, str], Tuple[pd.Timestamp, pd.Timestamp]],
                               n: int,
                               unit: str,
                               filepath: str = None,
                               **kwargs) -> None:
    
    color = kwargs.get('color', COLORS['GREEN'])
    fps   = kwargs.get('fps', 2)

    if unit == 's':
        type_name = 'seconds'
    elif unit == 'm':
        type_name = 'minutes'
    elif unit == 'h':
        type_name = 'hours'
    else:   
        type_name = 'milliseconds'

    name = 'artists' if 'ARTIST' in column.upper() else 'tracks'

    if time_range == 'ALL':
        start_date = pd.to_datetime(df_input['endTime'].min()).date() + pd.Timedelta(days=1)
        end_date   = pd.to_datetime(df_input['endTime'].max()).date() + pd.Timedelta(days=1)
        time_range = (start_date, end_date)
    elif isinstance(time_range, str):
        raise ValueError("Time range must be a tuple of strings or a tuple of pandas Timestamps.")
    
    frames = pd.date_range(start=time_range[0], end=time_range[1], freq='D')
    
    # Create animation
    animator = FuncAnimation(fig, partial(plot_hist_top_n, df_input=df_input, column=column, n=n, 
                                          unit=unit, previous_day= True, color=color), 
                             frames=frames)
    
    # Save the animation
    filepath = filepath if filepath else f'top_{n}_{name}_in_{type_name}.gif'
    filepath = f'{filepath}.gif' if not filepath.endswith('.gif') else filepath

    writergif = PillowWriter(fps=fps) 
    animator.save(filepath, writer=writergif)

create_gif_top_n_evolution(df, 'TRACK', 'ALL', 
                           15, 'm', color=COLORS['GREEN'])