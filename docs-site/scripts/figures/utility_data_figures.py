"""Reproduce utility/data diagrams: python3 docs-site/scripts/figures/utility_data_figures.py.

Analytic weights and synthetic bars from the runnable module-page examples.
No market data or openquant runtime required.
"""
from _svg import Chart, THEMES

for theme_name, theme in THEMES.items():
    c = Chart(720, 270, theme, 'Adjusted EWMA weights at the third observation, span 3')
    c.text(28, 30, 'Span 3: normalize the weights available so far', color='text', size=18)
    for i, (value, weight) in enumerate([(3, .25), (6, .5), (12, 1)]):
        x = 145 + i * 215
        c.band(x-48, x+48, 190-120*weight, 190, color='accent', opacity=.8)
        c.text(x, 180-120*weight, str(weight), anchor='middle', color='text')
        c.text(x, 214, f'x = {value}', anchor='middle')
    c.rule(65, 190, 655, 190)
    c.text(360, 249, '(0.25 × 3 + 0.5 × 6 + 1 × 12) / 1.75 = 9', anchor='middle', color='text', size=15)
    c.save('util-ewma-weights', theme_name)

    c = Chart(720, 250, theme, 'Range volatility warm-up with a three-bar window')
    c.text(28, 30, 'Window 3: Yang-Zhang needs a previous close', color='text', size=18)
    for i in range(5):
        c.text(280+90*i, 65, f'bar {i}', anchor='middle')
    for row, (name, warmup) in enumerate([('Parkinson', 2), ('Garman-Klass', 2), ('Yang-Zhang', 3)]):
        y = 105+45*row
        c.text(28, y+4, name, color='text', size=15)
        for i in range(5):
            x = 280+90*i
            c.dot(x, y, r=11, color='rule' if i < warmup else 'accent')
    c.text(28, 235, 'Muted = NaN warm-up; accent = first full windows (valid synthetic inputs).')
    c.save('util-volatility-warmup', theme_name)

    c = Chart(720, 240, theme, 'Calendar alignment inserts a missing synthetic daily bar')
    c.text(28, 30, 'Calendar alignment: explicit gaps, no invented prices', color='text', size=18)
    for row, name in enumerate(['Cleaned', 'Aligned']):
        y = 100+65*row
        c.text(28, y+5, name, color='text', size=15)
        c.rule(200,y,640,y)
        for i in range(4):
            x=220+130*i
            if row == 0 and i == 2:
                continue
            c.dot(x,y,r=12,color='rule' if i == 2 else 'accent')
            c.text(x,y+31,'null / missing' if i == 2 else ['101','102.5','','104'][i], anchor='middle')
    for i in range(4):
        c.text(220+130*i,64,f'Jan {i+1}',anchor='middle')
    c.text(28, 226, 'Synthetic closes; January 2 duplicate already resolved by keeping the last row.')
    c.save('data-calendar-gap',theme_name)
