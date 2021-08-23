import time
import csv
import logging
import matplotlib
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import basinhopping
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages


matplotlib.rcParams['font.family'] = ['sans-serif', ]
matplotlib.rcParams['font.sans-serif'] = ['SimHei', ]


def read_moving_data(day_start="2020-01-01"):
    """
    :return: dict, {省份_日期：从武汉迁入人数 (单位: w，按照1迁徙指数=4.7w换算}
    """
    date_start = datetime.strptime(day_start, "%Y-%m-%d")
    day_list = []
    curve_list = []
    result = defaultdict(float)
    with open('data/武汉迁出省数据.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        curve = next(reader, None)
        for (iidx, item) in enumerate(curve):
            if iidx:
                cur_day = (date_start + timedelta(days=iidx-1)).strftime("%Y-%m-%d")
                day_list.append(cur_day)
                curve_list.append(float(item))

        for (ridx, row) in enumerate(reader):
            loc = row[0]
            if '内蒙古' in loc or '黑龙江' in loc:
                loc = loc[:3]
            else:
                loc = loc[:2]
            for (iidx, data) in enumerate(row[1:]):
                data = 0.0 if data == "" else float(data)
                data = curve_list[iidx] * data * 47000.0
                key = "{}_{}".format(
                    loc, day_list[iidx]
                )
                result[key] = data
    return result


def param_penalty(x, lrange, rrange, ratio):
    if lrange <= x <= rrange:
        return 0.0
    else:
        return min(abs(x-lrange), abs(x-rrange)) * ratio


def step1_eq(par: tuple, initial_cond, start_t, end_t, incr, fix_par):
    t = np.linspace(start_t, end_t, incr)
    N = fix_par[0]
    def funct(y, t):
        Si = y[0]
        Ei = y[1]
        Ii = y[2]
        Ri = y[3]
        beta, delta, gamma = par
        f0 = - beta * Ii * Si / N
        f1 = beta * Ii * Si / N - delta * Ei
        f2 = delta * Ei - gamma * Ii
        f3 = gamma * Ii
        return [f0, f1, f2, f3]

    # integrate------------------------------------
    ds = integrate.odeint(funct, initial_cond, t)
    return ds[:, 0], ds[:, 1], ds[:, 2], ds[:, 3], t

def step1_score(
    params: list, id_stage: list, rd_stage: list,
    start_time_stage, end_time_stage,
    intervals_stage, mindex_stage, y0, fix_par
):
    score_sum = 0.0
    numstage = len(id_stage)
    beta_stage = params[:-2]
    delta, gamma = params[-2:]
    iscore_sum, rscore_sum = 0.0, 0.0
    for stagei in range(numstage):
        parms = (beta_stage[stagei], delta, gamma)
        Id = id_stage[stagei]
        Rd = rd_stage[stagei]
        start_time = start_time_stage[stagei]
        end_time = end_time_stage[stagei]
        intervals = intervals_stage[stagei]
        mindex = mindex_stage[stagei]
        F0, F1, F2, F3, T = step1_eq(
            parms, y0, start_time, end_time, intervals, fix_par
        )
        # b.Pick of Model Points to Compare
        Im = [F2[index] for index in mindex]
        Rm = [F3[index] for index in mindex]
        # c.Score Difference between model and data points
        ss = lambda data, model: ((data - model) ** 2).sum()
        iscore_sum += ss(np.array(Id), np.array(Im))
        rscore_sum += ss(np.array(Rd), np.array(Rm))
        score_sum += ss(np.array(Id), np.array(Im)) + \
                     ss(np.array(Rd), np.array(Rm))
        y0 = [F0[-1], F1[-1], F2[-1], F3[-1]]
        # force beta be greater than 0
        # score_sum += param_penalty(beta_stage[stagei], 0, 1000000, 50000000)

    return score_sum


def print_fun(x, f, accepted):
    logging.info("at minimum %.4f accepted %d" % (f, int(accepted)))


class RandomDisplacementBounds(object):
    """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
        Modified! (dropped acceptance-rejection sampling for a more specialized approach)
    """
    def __init__(self, xmin: list, xmax: list, stepsize: list):
        self.xmin = np.array(xmin, np.float)
        self.xmax = np.array(xmax, np.float)
        self.stepsize = np.array(stepsize, np.float)

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds """
        min_step = np.maximum(self.xmin - x, -self.stepsize)
        max_step = np.minimum(self.xmax - x, self.stepsize)

        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step

        return xnew


def step1_fit_model(
    Td, Id, Rd, y0,
    ini_rates: list,
    loc, fix_par, split: list
):
    """
    :param split: start or end points of each period, len(split) == numstage + 1
    :return:
    """
    start_time_stage = []
    end_time_stage = []
    intervals_stage = []
    mindex_stage = []
    id_stage = []
    rd_stage = []
    numstage = len(split) - 1
    for stagei in range(numstage):
        Td_thisstage = Td[split[stagei]:split[stagei+1]+1]
        mt = np.linspace(split[stagei], split[stagei+1],
                         (len(Td_thisstage) - 1) * 5 + 1)
        findindex = lambda x: np.where(mt >= x)[0][0]
        mindex = list(map(findindex, Td_thisstage))

        start_time_stage.append(split[stagei])
        end_time_stage.append(split[stagei+1])
        intervals_stage.append((len(Td_thisstage) - 1) * 5 + 1)
        mindex_stage.append(mindex)

        id_stage.append(Id[split[stagei]:split[stagei+1]+1])
        rd_stage.append(Rd[split[stagei]:split[stagei+1]+1])

    # bnds = [(0, None) for _ in range(numstage+2)]
    bnds = []
    for stagei in range(numstage + 2):
        bnds.append((0, 1))
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'args': (
            id_stage, rd_stage,
            start_time_stage, end_time_stage,
            intervals_stage, mindex_stage, y0, fix_par
        ),
        'bounds': bnds
    }

    args_min = [0 for _ in range(numstage + 2)]
    args_max = []
    for _ in range(numstage):
        args_max.append(10)
    args_max.append(3)
    args_max.append(3)
    args_step = []
    for stagei in range(numstage):
        if stagei == 0:
            args_step.append(0.5)
        elif stagei == numstage - 1:
            args_step.append(0.005)
        else:
            args_step.append(0.2)
    args_step.append(0.2)
    args_step.append(0.2)

    # print("args_min: ", args_min)
    # print("args_max: ", args_max)
    # print("args_step: ", args_step)

    bounded_step = RandomDisplacementBounds(
        xmin=args_min,
        xmax=args_max,
        stepsize=args_step
    )

    answ = basinhopping(
        func=step1_score,
        x0=np.array(ini_rates, np.float),
        minimizer_kwargs=minimizer_kwargs,
        niter=20,
        callback=print_fun,
        take_step=bounded_step,
        disp=False
    )

    bestrates = answ.x
    return bestrates, mindex_stage


def read_population(population_path):
    pop_data = defaultdict(int)
    with open(population_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            pop = int(row[3])
            loc = row[1]
            pop_data[loc] = pop
    return pop_data


def read_disease_data(province_disease_file, wuhan_disease_file):
    # 读取各省及武汉市确诊数据
    disease_cusum = defaultdict(float)
    with open(province_disease_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for (ridx, row) in enumerate(reader):
            if ridx:
                raw_date = row[4]
                loc = row[5]
                y, m, d = raw_date.split('/')
                m = '0' + m if len(m) < 2 else m
                d = '0' + d if len(d) < 2 else d
                date = '{}-{}-{}'.format(y, m, d)
                for (nidx, name) in enumerate(['累计确诊', '累计治愈',
                                              '累计死亡', '累计疑似']):
                    key = "{}_{}_{}".format(date, loc, name)
                    disease_cusum[key] = float(row[nidx])


    with open(wuhan_disease_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for (iidx, row) in enumerate(reader):
            if iidx:
                raw_date = row[4]
                city = row[6]
                y, m, d = raw_date.split('/')
                m = '0' + m if len(m) < 2 else m
                d = '0' + d if len(d) < 2 else d
                date = '{}-{}-{}'.format(y, m, d)
                for (nidx, name) in enumerate(['累计确诊', '累计治愈',
                                              '累计死亡', '累计疑似']):
                    key = "{}_{}_{}".format(date, city, name)
                    disease_cusum[key] = float(row[nidx])

    return disease_cusum


def fitting_initialization(disease_cusum,
                           pop_data):
    """
    获取拟合需要用到的初始值
    Args:
        disease_cusum: 各地疫情累计数据
        pop_data:

    Returns:

    """
    move_in_all = read_moving_data('2020-01-01')

    alltd = []
    allid = []
    allrd = []
    ally0 = []
    all_fixpara = []

    for loc in all_split:
        movein_start = '2020-01-09'
        movein_date_start = datetime.strptime(movein_start, '%Y-%m-%d')

        fit_start = all_fit_start[loc]
        fit_date_start = datetime.strptime(fit_start, '%Y-%m-%d')

        movein_windows = max(14, (fit_date_start-movein_date_start).days)

        fit_window = all_split[loc][-1]
        Td = []
        Id = []
        Rd = []
        imported = 0.0
        for didx in range(movein_windows):
            date = movein_date_start + timedelta(days=didx)
            day = date.strftime("%Y-%m-%d")
            key = "{}_{}".format(loc, day)
            movein = move_in_all[key]
            for name in ['累计确诊', '累计死亡', '累计治愈']:
                key = "{}_{}_{}".format(day, '武汉市', name)
                reported = disease_cusum[key]
                imported += movein * reported / 0.05 / 11080000.0

        for i in range(fit_window):
            cur_day = fit_date_start + timedelta(days=i)
            day = cur_day.strftime('%Y-%m-%d')
            Td.append(float(i))
            data_key = "{}_{}_{}".format(day, loc, '累计确诊')
            confirmed = disease_cusum[data_key]

            cured = disease_cusum['{}_{}_{}'.format(day, loc, '累计治愈')]
            death = disease_cusum['{}_{}_{}'.format(day, loc, '累计死亡')]
            Rd.append(cured + death)
            Id.append(confirmed - cured - death)

        N = pop_data[loc]
        E0 = imported
        I = disease_cusum["{}_{}_{}".format(fit_start, loc, '累计确诊')]
        R0 = disease_cusum["{}_{}_{}".format(fit_start, loc, '累计治愈')] + \
             disease_cusum["{}_{}_{}".format(fit_start, loc, '累计死亡')]
        I0 = I - R0
        S0 = N - E0 - I0 - R0  # initial population
        y0 = [S0, E0, I0, R0]  # initial condition vector
        alltd.append(Td)
        allid.append(Id)
        allrd.append(Rd)
        ally0.append(y0)
        all_fixpara.append((N, ))

    return alltd, allid, allrd, ally0, all_fixpara


def fitting(alltd, allid, allrd, ally0, all_fixpara):
    results = []
    with PdfPages(pdfname) as pdf:
        for (lidx, loc) in enumerate(all_split):
            fit_start = all_fit_start[loc]
            split = all_split[loc]

            loc_start = time.time()
            Td = alltd[lidx]
            Id = allid[lidx]
            Rd = allrd[lidx]
            y0 = ally0[lidx]
            fix_par = all_fixpara[lidx]

            logging.info("fit {} starts".format(loc))
            rates_step1, mindex_stage = step1_fit_model(
                Td=Td,
                Id=Id,
                Rd=Rd,
                y0=y0,
                # ini_rates=[1, 0.1, 0.01, 0.2, 0.2],
                ini_rates=[1, 0.1, 0.2, 0.2],
                loc=loc,
                fix_par=fix_par,
                split=split
            )

            result = [loc]
            result.extend(rates_step1)
            results.append(result)

            loc_end = time.time()
            logging.info("fit {} finishes".format(loc))
            logging.info("TIME: {:.2f}s".format(loc_end - loc_start))

            beta_stage = rates_step1[:-2]
            delta, gamma = rates_step1[-2:]

            # calculate the stage 2 model and visualize
            T_stage, Im_stage, Rm_stage = [], [], []
            numstage = len(mindex_stage)
            y0 = ally0[lidx]
            for stagei in range(numstage):
                beta = beta_stage[stagei]
                start_time = split[stagei]
                end_time = split[stagei+1]
                period_days = end_time - start_time
                intervals = period_days * 5 + 1
                F0, F1, F2, F3, T = step1_eq((beta, delta, gamma), y0,
                                                start_time, end_time,
                                                intervals, fix_par)
                T_stage.extend(T)
                Im_stage.extend(F2)
                Rm_stage.extend(F3)
                y0 = [F0[-1], F1[-1], F2[-1], F3[-1]]

            # Plot Solution to System Stage 1
            plt.figure(figsize=(15, 15))
            plt.plot(T_stage, Im_stage, 'b-', Td, Id, 'go')
            plt.xlabel('days')
            plt.ylabel('population')
            title_string = '{} 现存确诊 \n'.format(loc)
            for stagei in range(numstage):
                stage_start = datetime.strptime(fit_start, "%Y-%m-%d") + timedelta(days=split[stagei]-1)
                stage_end = datetime.strptime(fit_start, "%Y-%m-%d") + timedelta(days=split[stagei+1]-1)

                stage_start = stage_start.strftime("%Y/%m/%d")
                stage_end = stage_end.strftime("%Y/%m/%d")
                title_string += "第{}阶段：{}-{} beta: {} delta: {} gamma: {}\n".format(
                    stagei, stage_start, stage_end,
                    round(beta_stage[stagei], 4), round(delta, 4), round(gamma, 4)
                )
            plt.title(title_string, fontdict={'size': 18})
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(15, 15))
            plt.plot(T_stage, Rm_stage, 'b-', Td, Rd, 'go')
            plt.xlabel('days')
            plt.ylabel('population')

            title_string = '{} 累计治愈+死亡\n'.format(loc)
            for stagei in range(numstage):
                stage_start = datetime.strptime(fit_start, "%Y-%m-%d") + timedelta(days=split[stagei]-1)
                stage_end = datetime.strptime(fit_start, "%Y-%m-%d") + timedelta(days=split[stagei+1]-1)

                stage_start = stage_start.strftime("%Y/%m/%d")
                stage_end = stage_end.strftime("%Y/%m/%d")
                title_string += "第{}阶段：{}-{} beta: {} delta: {} gamma: {}\n".format(
                    stagei, stage_start, stage_end,
                    round(beta_stage[stagei], 4), round(delta, 4), round(gamma, 4)
                )

            plt.title(title_string, fontdict={'size': 18})
            pdf.savefig()
            plt.close()

            print("已处理完成第{}个省份：{} 还有{}个省份待处理 ".format(
                lidx,
                loc,
                len(all_split) - lidx - 1
            ))

    with open(csvname, 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


def main():
    # 读取各省人口数据
    pop_data = read_population(population_path)
    disease_cusum = read_disease_data(province_disease_file, wuhan_disease_file)
    alltd, allid, allrd, ally0, all_fixpara = fitting_initialization(
        disease_cusum,
        pop_data
    )

    fitting(alltd, allid, allrd, ally0, all_fixpara)


if __name__ == "__main__":
    logname = "2阶段_关注度60_关注度60后7日.log"

    # setting config
    logging.basicConfig(
        filename=logname,
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(pathname)s[line:%(lineno)d]: %(message)s'
    )

    province_list = [
        '黑龙江', '吉林', '辽宁', '内蒙古',
        '北京', '天津', '河北', '山东', '山西', '河南',
        '陕西', '甘肃', '宁夏', '新疆', '青海',
        '浙江', '江西', '安徽', '江苏', '上海', '湖南',
        '广东', '福建', '广西', '海南',
        '重庆', '四川', '云南', '贵州']

    all_fit_start = defaultdict(str)
    all_split = defaultdict(list)

    policy_df = pd.read_csv("data/政策时间.csv")

    for keyword in ['勤洗手', '早隔离', '测体温', '交通管制', '居家隔离', '戴口罩']:
        keyword_policy_df = policy_df[policy_df["政策"] == keyword]
        for loc in province_list:
            high_attention_time = datetime.strptime(keyword_policy_df[keyword_policy_df["地点"] == loc].reset_index(drop = True).loc[0,"高关注度时间"],'%Y-%m-%d')
            stage12_interval_time = 7
            all_fit_start[loc] = (high_attention_time - timedelta(days=7)).strftime("%Y-%m-%d")#各省的起始拟合时间设置为高关注度时间-7days
            all_split[loc] = [0, 7, 14]  #阶段划分始终为[0,7,14]

        for delay_window in range(0, 1):
            pdfname = "0520/关注度60前7日_关注度60_关注度60后7日_{}_delay{}.pdf".format(
                keyword, delay_window)  # 拟合曲线会被画到这个pdf中
            csvname = "0520/关注度60前7日_关注度60_关注度60后7日_{}_delay{}.csv".format(
                keyword, delay_window)  # 各省拟合结果会被保存在这个csv中
            split_csvname = "0520/阶段划分_关注度60前7日_关注度60_关注度60后7日_{}_delay{}.csv".format(
                keyword, delay_window)

            population_path = "data/Pop_States.csv"
            province_disease_file = 'data/wuhan_2019_ncov_processed.csv'
            wuhan_disease_file = 'data/wuhan_2019_ncov_WuhanCity.csv'

            with open(split_csvname, 'w', newline="") as csvfile:
                writer = csv.writer(csvfile)
                for province in all_split:
                    split = all_split[province]
                    start = all_fit_start[province].replace('-', '/')

                    writer.writerow([province] + split + [start])

            main()
