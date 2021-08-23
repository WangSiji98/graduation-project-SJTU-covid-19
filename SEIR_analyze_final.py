import csv
import xlwt
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from collections import defaultdict
from scipy.stats import pearsonr

import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 换成你安装的字体
matplotlib.rcParams['font.family'] = ['sans-serif', ]
matplotlib.rcParams['font.sans-serif'] = ['SimHei', ]


def get_ranged_date(start, end):
    """
    return a range of dates between start and end (both closed)
    FIXME: the range must in 2020
    :param start:
    :param end:
    :return:
    """
    y1, m1, d1 = [int(item) for item in start.split('-')]
    y2, m2, d2 = [int(item) for item in end.split('-')]
    result = []
    day_per_month = [31, 29, 31, 30, 31, 30, 31, 31]  # year 2020
    for m in range(m1, m2+1):
        dstart = d1 if m == m1 else 1
        dend = d2 if m == m2 else day_per_month[m-1]
        for d in range(dstart, dend+1):
            printm = '0' + str(m) if m < 10 else str(m)
            printd = '0' + str(d) if d < 10 else str(d)
            result.append('2020-{}-{}'.format(printm, printd))
    return result


class SEIRResultV2(object):
    def __init__(self, province, betas: list, delta, gamma,
                 allmoving=[], alldisease=[]):
        self.province = province
        self.betas = betas
        self.delta = delta
        self.gamma = gamma
        self.allmoving = allmoving
        self.alldisease = alldisease

    def getList(self):
        # return [self.province, self.c, self.beta, self.delta, self.gamma]
        result = [self.province]
        result.extend(self.betas)
        result.append(self.delta)
        result.append(self.gamma)
        return result

    def getParameterList(self):
        result = []
        result.extend(self.betas)
        result.append(self.delta)
        result.append(self.gamma)
        return result


def write_excel_xls(value, path, sheet_name='sheet1'):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿


def read_focusing_data():
    phrase_list = []
    datas = defaultdict(list)
    if focus_type == 'Absolute':
        with open(phrase_data, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                key = "_".join(row[:3]) # 日期, 省份, 词条
                if row[2] not in phrase_list:
                    phrase_list.append(row[2])
                datas[key][0] += int(row[3])
    elif focus_type == 'Relative':
        with open(phrase_data, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                key = "_".join(row[:3]) # 日期, 省份, 词条
                if row[2] not in phrase_list:
                    phrase_list.append(row[2])
                datas[key] = [float(row[3]), float(row[5])]
    elif focus_type == 'GlobalRelative':
        fin = open("data/place_time.clean", 'r')
        globalfocus = defaultdict()
        for line in fin.readlines():
            key, value = line.split('\t')
            globalfocus[key] = value

        with open(phrase_data, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                key = "_".join(row[:3])  # 日期, 省份, 词条
                if row[2] not in phrase_list:
                    phrase_list.append(row[2])
                datas[key] = [float(row[3])]

        for key in datas.keys():
            items = key.split('_')
            sum_key = items[1] + '#' + items[0]
            datas[key].append(float(globalfocus[sum_key]))

    return datas, phrase_list


def read_fitting_model():
    province_result = defaultdict(SEIRResultV2)

    with open(modelpath, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # 地点 beta delta gamma
            loc = row[0]
            betas = [float(item) for item in row[1:-2]]
            delta = float(row[-2])
            gamma = float(row[-1])

            province_result[loc] = SEIRResultV2(
                loc,
                betas,
                delta,
                gamma
            )
    return province_result


def analyze(datas, phrase_list, province_result,
            focus_selected_range: dict):
    """
    Args:
        datas:
        phrase_list:
        province_result:
        focus_selected_range: dict，表示为每个省份选取哪个时段的关注度进行分析。
            key为省份，value为开始和结束日期
            e.g. {'广东': ['2020-01-23', '2020-01-28']}

    Returns:

    """
    results = []
    for m in range(m_start, m_end):
        for province in province_list:
            if province not in all_fit_start:
                continue
            seirstartdate = datetime.strptime(
                all_fit_start[province], "%Y-%m-%d")

            stage2startdate = seirstartdate + \
                              timedelta(days=int(all_split[province][1]))

            if not fixed_t0:
                stage2startdate += timedelta(days=int(delay))

            focus_selected_range[province] = [
                (stage2startdate - timedelta(days=(m + window_size))).strftime('%Y-%m-%d'),
                (stage2startdate - timedelta(days=m)).strftime('%Y-%m-%d')
            ]

        # for loc in focus_selected_range:
        #     print("delay: {} m: {}".format(delay, m))
        #     print(loc, focus_selected_range[loc])

        # 将相关度参数写入excel文件
        """
        alldir = "m={}_关注度与第{}阶段模型参数_所有词条_0515".format(
            m, model_stage
        )

        if exclude_hubei:
            alldir += "_除湖北"
        if not os.path.exists(alldir):
            os.mkdir(alldir)

        selecteddir = "m={}_关注度与第{}阶段模型参数_相关性明显词条_0515".format(
            m, model_stage
        )
        if exclude_hubei:
            selecteddir += "_除湖北"
        if not os.path.exists(selecteddir):
            os.mkdir(selecteddir)
        """

        oneresult = []
        best_result = {'pearson': 0.0, 'sig': 0.99}

        # 窗口内每个关键词在各省的关注度
        tmp_focus = dict()
        for (key, value) in datas.items():
            date, loc, phr = key.split("_")
            if loc in focus_selected_range:
                focus_start, focus_end = focus_selected_range[loc]
                date_list = get_ranged_date(focus_start, focus_end)
                if date in date_list:
                    new_key = loc + "_" + phr  # 省份_词条
                    if len(value) == 1:
                        if new_key not in tmp_focus:
                            tmp_focus[new_key][0] = value[0]
                        else:
                            tmp_focus[new_key][0] += value[0]
                    else:
                        if new_key not in tmp_focus:
                            tmp_focus[new_key] = value
                        else:
                            tmp_focus[new_key][0] += value[0]
                            tmp_focus[new_key][1] += value[1]

        focus_data = defaultdict(int)
        for (key, value) in tmp_focus.items():
            if len(value) == 1:
                focus_data[key] = tmp_focus[key][0]
            else:
                focus_data[key] = tmp_focus[key][0] / tmp_focus[key][1]

        # 对齐关注度和斜率
        focuses = []
        slopes = []
        locations = []
        for loc in province_result:
            if exclude_hubei and loc in ['湖北', '西藏']:
                continue
            focuses.append(focus_data[loc + "_" + keyword])  # 窗口长度为w天
            if args.residual:
                slopes.append(
                    province_result[loc].betas[model_stage - 2] -
                    province_result[loc].betas[model_stage - 1]
                )
            else:
                slopes.append(province_result[loc].betas[model_stage - 1])
            locations.append(loc)

        pearson, sig = pearsonr(focuses, slopes)
        if sig < best_result['sig']:
            best_result['pearson'] = pearson
            best_result['sig'] = sig

        pearson = best_result['pearson']
        sig = best_result['sig']
        # if sig < 0.1:
        # plt.figure(figsize=(12, 9))
        plt.figure()
        ax = plt.subplot()
        ax.scatter(focuses, slopes)
        ax.set_title("「{}」关注度与政策起效前后beta差值的关系 \n r: {} p: {}\n".format(
            keyword, round(pearson, 3), round(sig, 6)
        ))
        ax.set(xlabel='关注度', ylabel='SEIR模型中易感者转化为潜伏者速率')
        for idx in range(len(focuses)):
            loc = locations[idx]
            focus = focuses[idx]
            slope = slopes[idx]
            ax.annotate(loc, xy=(focus, slope), fontsize=8)
        pdf.savefig()
        plt.close()
        oneresult.append(pearson)
        oneresult.append(sig)

        # 将相关度参数写入excel文件
        """
        filename = os.path.join(alldir, '{}.xls'.format(phrase))
        values = [
            ['地点', '平均相对关注度', 'beta'],
        ]
        for (f, s, l) in zip(focuses, slopes, locations):
            values.append([l, f, s])
        write_excel_xls(values, filename)

        if sig < 0.1:
            filename = os.path.join(selecteddir, '{}.xls'.format(phrase))
            values = [
                ['地点', '平均相对关注度', 'beta1'],
            ]
            for (f, s, l) in zip(focuses, slopes, locations):
                values.append([l, f, s])
            write_excel_xls(values, filename)
        """
        results.append(oneresult)

    # results = np.stack(results, axis=1)
    # indexs = []
    # for phrase in phrase_list:
    #     indexs.append(phrase + "_pearsonr")
    #     indexs.append(phrase + "_sig")
    #
    # columns = [m for m in range(m_start, m_end)]
    # resultsdf = pd.DataFrame(results, index=indexs, columns=columns)
    # resultsdf.to_excel(resultsxls)

    # resultsdf.to_tsv(resultstsv, sep='\t')


def main():
    focus_data, phrase_list = read_focusing_data()
    province_model = read_fitting_model()

    # for key in province_model:
    #     if key not in all_fit_start:
    #         del province_model[key]
    new_province_model = defaultdict(SEIRResultV2)
    for key in province_model:
        if key in all_fit_start:
            new_province_model[key] = province_model[key]
    province_model = new_province_model

    analyze(focus_data, phrase_list,
            province_model, focus_selected_range)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--focus_type", type=str, default='Relative')
    parser.add_argument("--fixed_t0", action='store_true')
    parser.add_argument("--residual", action='store_true')

    args = parser.parse_args()

    assert args.focus_type in ['Absolute', 'Relative', 'GlobalRelative']

    key1 = "fixedt0" if args.fixed_t0 else "variedt0"
    key2 = "residual" if args.residual else "absolute"

    dirname = "0520"

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # 少湖北 西藏
    province_list = [
        '黑龙江', '吉林', '辽宁', '内蒙古',
        '北京', '天津', '河北', '山东', '山西', '河南',
        '陕西', '甘肃', '宁夏', '新疆', '青海',
        '浙江', '江西', '安徽', '江苏', '上海', '湖南',
        '广东', '福建', '广西', '海南',
        '重庆', '四川', '云南', '贵州']

    phrase_data = 'data/freq_total.tsv'

    focus_type = args.focus_type
    fixed_t0 = args.fixed_t0
    window_size = 7
    m_start = 0
    m_end = 1

    pdf_name = "{}/政策类词条相关性_{}_{}_w{}.pdf".format(
        dirname, key2, args.focus_type, window_size
    )

    with PdfPages(pdf_name) as pdf:
        for keyword in ['测体温', '戴口罩', '交通管制', '居家隔离',  '勤洗手', '早隔离']:
            for delay in range(0, 1):
                modelpath = "0520/关注度60前7日_关注度60_关注度60后7日_{}_delay{}.csv".format(
                    keyword, delay
                )
                splitcsvname = '0520/阶段划分_关注度60前7日_关注度60_关注度60后7日_{}_delay{}.csv'.format(
                    keyword, delay)
                # resultsxls = "关注度与参数相关度_delay{}.xls".format(delay)

                exclude_hubei = 1  # 是否只分析湖北省之外的模型
                model_stage = 2  # 选用第几阶段的模型参数进行分析，starts from 1

                # 筛选位于所有省份高关注度时间中位数正负四天之内的结果
                all_fit_start = defaultdict()
                all_split = defaultdict()
                items = []  # [省份，阶段划分，开始日期，时间戳]
                with open(splitcsvname, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        item = [row[0], row[1:-1]]
                        start = row[-1].replace('/', '-')
                        item.append(start)
                        item.append(
                            (datetime.strptime(start, "%Y-%m-%d") -
                            datetime.strptime("2020-01-01", "%Y-%m-%d")).days
                        )
                        items.append(item)
                items = sorted(items, key=lambda x: x[3])
                mid_timestamp = items[int(len(items) / 2)][3]
                selected_items = [item for item in items if item[3] in range(mid_timestamp-4, mid_timestamp+5)]
                for item in selected_items:
                    province = item[0]
                    all_split[province] = item[1]
                    all_fit_start[province] = item[2]

                print("keyword: {} province after selected: {}".format(
                    keyword, len(selected_items)
                ))

                focus_selected_range = dict()
                main()