# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import Time_Index as ti
    from misc import PALETTE, lim_margin
else:
    import utils.Time_Index as ti
    from utils.misc import PALETTE, lim_margin

from datetime import datetime, timedelta
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
'''
default_font = 'Times New Roman'
default_font_kor = 'NanumMyeongjo'
#plt.rcParams['font.family'] = default_font
#plt.rcParams['mathtext.fontset'] = 'custom'
#plt.rcParams['mathtext.rm'] = f'{default_font}'
#plt.rcParams['mathtext.it'] = f'{default_font}:italic'
#plt.rcParams['mathtext.bf'] = f'{default_font}:bold'

sty_figure = {'facecolor':'w', 'edgecolor':'k'}
sty_title = {'fontsize': 18, 'pad': 10, 'fontweight': 'bold'}
sty_plot = {'linewidth': 0.8}
sty_axis_tick = {'direction':'in', 'length':3, 'labelsize':14, 'pad':5,
        'bottom': True, 'top': True, 'left': True, 'right': True}
sty_vhlines = {'lw': 1.5}
sty_label = {'labelpad':16, 'fontsize':15, 'va': 'center', 'fontweight': 'bold'}
sty_legend = {'edgecolor': 'k', 'facecolor': 'w', 'framealpha': 0.8, 'labelspacing': 0.7,
        'handletextpad': 0.6, 'fontsize': 12, 'columnspacing': 1.0,
        'borderpad': 0.6, 'borderaxespad': 0, 'prop': {'family': default_font_kor}}
sty_legend_ext = {**sty_legend, 'loc': 'upper left', 'handlelength': 2.0,
        'bbox_to_anchor': (1.05, 1)}

sty_legend_sch = {**sty_legend, 'loc': 'lower left', 'ncol': 2, 'handlelength': 1.5,}
sty_text = {'va': 'bottom', 'fontsize': 12, 'fontweight': 'bold',
        'bbox': dict(pad=20, ec='none', fc='none')}
sty_grid = {'color': 'k', 'alpha': 0.2}
'''

def plot_ctrl_result(model, res, save_loc='Results/'):

    params = res['params']

    cool_color = ['royalblue', 'green', 'red', 'black']
    cool_name = ['제빙', '축단', '병렬', '냉단']
    heat_color = ['#9ad1f5', '#e49af5']

    fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True, **sty_figure,
            gridspec_kw={'height_ratios': [7, 1]})
    ax, ax_sch = axs.ravel()
    
    ra_temp = res['table']['RA']
    time_str_range = ra_temp.index
    time_range = [ti.string_to_time(_) for _ in time_str_range]
    ax.plot(ra_temp[:model.TS[0]], 'o-', color=PALETTE[4], lw=1.2, zorder=10,
            mfc='none', label='과거 측정값($T_{RA,measured}$)')

    df = res['df']
    after_optimal = False
    for idx, column in enumerate(df.columns):
        optimal = column == ti.time_to_string(*res['time'])
        if optimal:
            after_optimal = True
        ax.plot(
            df.loc[:, column],
            ('o-' if optimal else '-'),
            color=(PALETTE[3] if (after_optimal and res['success']) else PALETTE[1]),
            mfc='none',# ms=0.5,
            alpha=(1.0 if optimal else 0.4),
            label=('목표 도달 실패' if (not optimal and idx==0) \
                    else '최종 제어 $(T_{RA,predicted})$' \
                    if optimal else None),
            lw=(1.0 if optimal else 0.8)
        )

    y_lim = ax.get_ylim()

    ax.hlines(res['target temp'], -10, sum(model.TS)+10, \
            color=PALETTE[2], label='목표 시간 & 온도', **sty_vhlines)
    ax.vlines(model.TS[0]+params['target_ts'], 0, 50, \
            color=PALETTE[2], **sty_vhlines)
    con_name_kor = '기동' if params['is_start'] else '정지'
    ax.vlines((datetime(*res['time'])-datetime(*time_range[0]))/ti.TS, 0, 40, \
            color=PALETTE[3], ls=(0,(6,5.5)), label=f'공조기 최적 {con_name_kor}시점', \
            **sty_vhlines)
    ax.text(s=f'{res["time"][3]}:{res["time"][4]:02}',
            x=(datetime(*res['time'])-datetime(*time_range[0]))/ti.TS \
                + (0.4 if params['is_start'] else -0.4),
            y=y_lim[0]+(y_lim[1]-y_lim[0])*0.02, color=PALETTE[3],
            ha=('left' if params['is_start'] else 'right'), **sty_text)
    
    ax.set_xlim(time_str_range[0], time_str_range[-1])
    ax.set_ylim(*y_lim)
    ax_sch.set_ylim(0, 1)

    start_time_dt = datetime(*res['start time'])+ti.TS
    xticklabels = [time_str_range[0]] \
            + [ti.datetime_to_string(start_time_dt)] \
            + [ti.datetime_to_string(start_time_dt + ti.TS*params['target_ts'])]
    
    ax.set_xticks(range(sum(model.TS)))
    ax.set_xticklabels(
        [f'{ti.string_to_time(_)[3]}:{ti.string_to_time(_)[4]:02}' if _ in xticklabels else '' for _ in ra_temp.index]
    )
    # ax.set_xticks(
    #     [_ for _ in range(sum(model.TS)) if time_range[_][4]==0]
    # )
    # ax.set_xticklabels(
    #     [f'{ti.string_to_time(_)[3]:02}:{ti.string_to_time(_)[4]:02}' if _ in xticklabels else '' for _ in ra_temp.index]
    # )

    day = time_range[0]
    ax.set_title(f'{day[0]:02}-{day[1]:02}-{day[2]:02}', **sty_title)
    ax.set_ylabel('RA Temperature [°C]', **sty_label)

    aux_x = range(sum(model.TS))
    aux_trans = ax_sch.get_xaxis_transform()
    if params['is_cool']:
        # patch_list = [mpatches.Patch(color=cool_color[_], alpha=0.3) for _ in range(4)]
        # patch_label = [c+' 운전' for c in cool_name]

        # for j in range(4):
        #     ax_sch.fill_between(aux_x, 0, 1, where=(res['table']['COOL OP']==j),
        #             color=cool_color[j], ec='none',
        #             alpha=0.3, transform=aux_trans)
        patch_list = mpatches.Patch(color=heat_color[0])
        patch_label = '냉열원 기동상태'

        ax_sch.fill_between(aux_x, 0, 1, where=(res['table']['COOL OP']!=0),
                fc=heat_color[0], ec='none', alpha=1.0, transform=aux_trans)
    else:
        patch_list = mpatches.Patch(color=heat_color[1])
        patch_label = '온열원 기동상태'

        ax_sch.fill_between(aux_x, 0, 1, where=(res['table']['BOIL OP']!=0),
                fc=heat_color[1], ec='none', alpha=1.0, transform=aux_trans)

    ax.tick_params(**sty_axis_tick)
    ax_sch.tick_params(axis='x', **sty_axis_tick)
    ax_sch.set_yticks([])

    ax.grid(**sty_grid)
    ax_sch.grid(**sty_grid)

    ax.legend(**sty_legend_ext)
    leg_h, leg_l = ax.get_legend_handles_labels()
    leg_h.append(patch_list)
    leg_l.append(patch_label)
    ax.legend(leg_h, leg_l, **sty_legend_ext)

    # ax_sch.legend(patch_list, patch_label, **sty_legend_sch,
    #         bbox_to_anchor=(1.05, 0, leg.get_bbox_to_anchor()._bbox.width, 0), mode='expand')

    plt.tight_layout()

    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    #fig.savefig(f'{save_loc}{model.name}.jpg', bbox_inches='tight', dpi=300)
    plt.close()

# for wk in range(4):
#     optimal_control_test(datetime_to_tuple(datetime(2020, 8, 3)+timedelta(days=7)*wk)[:3], True)

