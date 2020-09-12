from app.signal_calc import SignalCalc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import numpy as np
import pandas as pd


def line_fig(date, y_axis, y_dict):
    """
    :param date: xaxis,
    :param y_label: yaxis label e.g. 'y2','y3'
    :param y_dict: line data
    :return: list
    """

    y_values = [list(k) for k in y_dict.values()]
    y_labels = [l for l in y_dict.keys()]
    y_data = np.array(y_values)

    fig_res = []
    for i in range(len(y_values)):
        fig = go.Scatter(
            x=date,
            y=y_data[i],
            mode='lines',
            marker=dict(opacity=0.8),
            name=y_labels[i],
            connectgaps=True,
            yaxis=y_axis,
        )

        fig_res.append(fig)
    return fig_res


class QuantFig:

    def __init__(self, df: pd.DataFrame, start_date: str = '', end_date: str = ''):

        self._increasing_color = '#FF5C5C'
        self._decreasing_color = '#46A346'

        df['color_label'] = self._decreasing_color
        df.loc[df.close < df.open, 'color_label'] = self._decreasing_color
        df.loc[df.close >= df.open, 'color_label'] = self._increasing_color
        df = df.sort_values(by='date')  # order old -> new

        self._df = df
        self._start_date = start_date
        self._end_date = end_date
        self._date = df['date']
        self._open = df['open']
        self._high = df['high']
        self._low = df['low']
        self._close = df['close']
        self._volume = df['volume']
        self._color_label = df['color_label']

        self._data = [self.add_bars()]
        self._label = ['Price']

        self._ta = SignalCalc(self._df)
        self._shapes = []

    def show(self):
        fig = make_subplots(shared_xaxes=True, vertical_spacing=0.02)

        data = self._data

        layout = go.Layout(
            autosize=True,
            height=800,
            # width=1280,
            yaxis=dict(
                domain=[0.41, 1],
                title=self._label[0],
            ),
            xaxis_rangeslider_visible=False,
            showlegend=False,
        )

        fig = go.Figure(data=data, layout=layout)

        t = 40
        margin = 2
        ysize = math.floor(t / (len(self._label) - 1))

        for i in range(1, len(self._label)):
            yax_name = 'yaxis' + str(i + 1)
            d_upbound = t - margin
            t -= ysize
            d_lowbound = t if t >= ysize else 0
            fig.update_layout(
                **{yax_name: dict(
                    domain=[d_lowbound / 100, d_upbound / 100],
                    title=self._label[i],
                )}
            )

        fig.update_xaxes(showline=True, linewidth=0, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, automargin=True)
        fig.update_layout(
            hovermode="x unified",
            template="gridon",
            margin=dict(
                l=10,
                r=30,
                b=10,
                t=10,
                pad=4
            ),
            paper_bgcolor="#f3f5f0",
        )

        for shape in self._shapes:
            fig.add_shape(shape)

        return fig

    def add_annot(self, annot):
        buy = None
        for _, data in annot[['date', 'text']].iterrows():
            date, text = data
            if text == 'Buy' or text == 'Sell':
                if text == 'Buy':
                    buy = date
                    self._shapes.append(dict(
                        type='line', x0=date, x1=date, y0=0, y1=1, xref='x', yref='paper',
                        line_width=1, line=dict(color="Orange")
                    ))
                else:
                    if buy is not None:
                        # self._shapes.append(dict(
                        #     type="rect", x0=buy, x1=date, y0=0, y1=1, xref='x', yref='paper',
                        #     line_width=1, line=dict(width=0), fillcolor="LightSkyBlue", opacity=0.5)
                        # )
                        buy = None
                    self._shapes.append(dict(
                        type='line', x0=date, x1=date, y0=0, y1=1, xref='x', yref='paper',
                        line_width=1, line=dict(color='RoyalBlue')
                    ))

    def reset_annot(self):
        self._shapes.clear()

    def add_bars(self):
        bar = go.Candlestick(
            x=self._date,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            increasing={'line': {'color': self._increasing_color}},
            decreasing={'line': {'color': self._decreasing_color}},
            yaxis='y',
            name=""
        )

        return bar

    def add_volume(self):
        len1 = len(self._label)
        ylabel = 'y' + str(len1 + 1)
        label = ['Volume']

        volume = go.Bar(
            x=self._date,
            y=self._volume,
            name="Volume",
            marker={'color': self._color_label},
            yaxis=ylabel
        )

        self._data = self._data + [volume]
        self._label = self._label + label

        return

    def add_ema(self, tp1=3, tp2=10):
        y_axis = 'y'

        y_dict = self._ta.ema(tp1=tp1, tp2=tp2)
        y_dict.pop('signal0')

        fig_res = line_fig(self._date, y_axis, y_dict)

        self._data = self._data + fig_res

        return

    def add_dualma(self, tp1=20, tp2=20):
        y_axis = 'y'

        y_dict = self._ta.dualma(tp1=tp1, tp2=tp2)
        y_dict.pop('signal0')

        fig_res = line_fig(self._date, y_axis, y_dict)

        self._data = self._data + fig_res

        return

    def add_macd(self):
        y_axis = 'y' + str(len(self._label) + 1)
        label = ['MACD']

        y_dict = self._ta.macd()
        y_dict.pop('signal0')

        # fig_res = line_fig(self._date, y_axis, y_dict)

        y_values = [list(k) for k in y_dict.values()]
        y_labels = [l for l in y_dict.keys()]
        y_data = np.array(y_values)
        fig_res = []
        for i in range(0, len(y_values)):
            if i == len(y_values)-1:
                fig = go.Bar(
                    x=self._date,
                    y=y_data[i],
                    name=y_labels[i],
                    marker={'color': "#bfbfbf"},
                    yaxis=y_axis,
                )
            else:
                fig = go.Scatter(
                    x=self._date,
                    y=y_data[i],
                    mode='lines',
                    name=y_labels[i],
                    connectgaps=True,
                    yaxis=y_axis,
                )
            fig_res.append(fig)

        self._data = self._data + fig_res
        self._label = self._label + label

        return

    def add_kdj(self):
        y_axis = 'y' + str(len(self._label) + 1)
        label = ['KDJ']

        y_dict = self._ta.kdj()
        y_dict.pop('signal0')

        fig_res = line_fig(self._date, y_axis, y_dict)

        self._data = self._data + fig_res
        self._label = self._label + label

        return

    def add_rsi(self):
        y_axis = 'y' + str(len(self._label) + 1)
        label = ['RSI']

        y_dict = self._ta.rsi()
        y_dict.pop('signal0')

        fig_res = line_fig(self._date, y_axis, y_dict)

        self._data = self._data + fig_res
        self._label = self._label + label

        return

    def addLines(self):

        return
