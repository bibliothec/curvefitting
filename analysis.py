# coding: UTF-8
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from numba import jit

# csvから読み取る関数
def read_csv(path):
    # ch0は光電子増倍管, ch1はピエゾ, ch2はエタロン
    df = pd.read_csv(path, sep=',', header=None,  names=['SampleNum', 'DataTime', 'ch0', 'ch1', 'ch2', 'Events'])
    # CSV上の余計なデータの削除
    # 不要な列の削除
    drop_col = ['SampleNum', 'DataTime', 'Events']
    df = df.drop(drop_col,  axis=1)
    # 不要な行の削除
    # 上のヘッダー情報を削除
    header_num = 8
    df = df.drop(range(header_num))
    # dfから文字列を削除したので、残りのdtypeをobjectからfloatに変換
    df = df.astype(float)
    # ピエゾの三角波の最大値からになるように、それ以前のデータを削除
    df_tmp = df.head(sampling_rate)
    max_index = df_tmp['ch1'].idxmax()
    df = df.drop(range(header_num, max_index))
    # indexの振り直し
    df = df.reset_index(drop=True)
    return df

# 整形したCSVを出力する関数
def export_shaped_csv():
    LonPon.to_csv(path_dir + 'new_LonPon.csv',  encoding = 'utf-8')
    LoffPon.to_csv(path_dir + 'new_LoffPon.csv',  encoding = 'utf-8')
    LonPoff.to_csv(path_dir + 'new_LonPoff.csv',  encoding = 'utf-8')
    LoffPoff.to_csv(path_dir + 'new_LoffPoff.csv',  encoding = 'utf-8')

# 目的関数
def objective_function( lambda_est, temperature, gas_density ):
    # 理論値αの配列
    alpha_theo = []
    for i in index:
        S2 = const_1/math.sqrt(temperature)
        S1 = (1/((lambda_est+((i-max_index)*delta_lambda))*(10**(-9))))-(1/lambda_0)
        S = math.exp((-1 * (S1 * S2) ** 2))
        alpha_i = 1 - math.exp((-1*const_2*(10**(gas_density))/(math.sqrt(temperature))) * S)
        alpha_theo.append(alpha_i)
    return alpha_theo

# λを取得(x軸)
def get_lambda_theo():
    # 理論値のλの配列(x軸の値になる)
    lambda_theo = []
    for i in index:
        lambda_i = tmp_center_pos + delta_lambda * (i - max_index)
        lambda_theo.append(lambda_i)
    return lambda_theo

# グラフ作成
def plot_graph():
    plt.figure(figsize = (10,6), facecolor='lightblue')
    plt.plot(lambda_theo, alpha_exp, color='red', label='α(exp)')
    plt.plot(lambda_theo, alpha_theo, color='blue', label='α(cal)')
    plt.xlabel("wavelength[nm]") 
    plt.ylabel("absorption rate") 
    plt.legend(loc = 'upper right')
    plt.savefig("output.png")
    # plt.show()


class PSO():
    # pso parametars
    # basic
    particle_num = 30
    unknown_num = 3
    alpha_theo = []
    nowscore = 10000000
    # 重み
    pbest_num = 2.5
    gbest_num = 1.7
    lambda_w = 0.4
    lambda_c1 = pbest_num
    lambda_c2 = gbest_num
    temperature_w = 0.7
    temperature_c1 = pbest_num
    temperature_c2 = gbest_num
    gas_density_w = 0.7
    gas_density_c1 = pbest_num
    gas_density_c2 = gbest_num
    # パラメータの最小値と最大値
    lambda_max = 696.5435
    lambda_min = 696.5425
    gas_density_max = 18
    gas_density_min = 15
    temperature_max = 600
    temperature_min = 200
    # パラメータのセッティング
    pbestscore = [0] * particle_num
    gbestscore = 100000000
    # 最終的なパラメータ
    lambda_v = [0.0]*particle_num
    lambda_x = [0.0]*particle_num
    lambda_pbest = [0.0]*particle_num
    lambda_gbest = 0
    temperature_v = [0.0]*particle_num
    temperature_x = [0.0]*particle_num
    temperature_pbest = [0.0]*particle_num
    temperature_gbest = 0
    gas_density_v = [0.0]*particle_num
    gas_density_x = [0.0]*particle_num
    gas_density_pbest = [0.0]*particle_num
    gas_density_gbest = 0
    # 初期値, コンストラクタ
    def __init__(self):
        for i in range(self.particle_num):
            # 位置と速度の初期化
            self.lambda_x[i] = random.uniform(self.lambda_min, self.lambda_max)
            self.temperature_x[i] = random.uniform(self.temperature_min, self.temperature_max)
            self.gas_density_x[i] = random.uniform(self.gas_density_min, self.gas_density_min)
            self.lambda_pbest[i] = self.lambda_x[i]
            self.temperature_pbest[i] = self.temperature_x[i]
            self.gas_density_pbest[i] = self.gas_density_x[i]
            v_keisu_tmp = 0.5
            self.lambda_v[i] = (self.lambda_max - self.lambda_min) * v_keisu_tmp * (random.random() - 0.5)
            self.temperature_v[i] = (self.gas_density_max - self.gas_density_min) * v_keisu_tmp * (random.random() - 0.5)
            self.gas_density_v[i] = (self.temperature_max - self.temperature_min) * v_keisu_tmp * (random.random() - 0.5)
            # 初期値でのスコアを得る、各粒子毎
            self.pbestscore[i] = self.getalphascore(self.lambda_x[i], self.temperature_x[i], self.gas_density_x[i])
            # 初期値でのglobalbestを得る
            if self.pbestscore[i] < self.gbestscore:
                self.gbestscore = self.pbestscore[i]
                self.lambda_gbest = self.lambda_pbest[i]
                self.temperature_gbest = self.temperature_pbest[i]
                self.gas_density_gbest = self.gas_density_pbest[i]
    def move(self):
        for i in range(self.particle_num):
            self.lambda_v[i] = self.lambda_w*random.random()*self.lambda_v[i] + self.lambda_c1*random.random()*(self.lambda_pbest[i]-self.lambda_x[i]) + self.lambda_c2*random.random()*(self.lambda_gbest-self.lambda_x[i])
            self.temperature_v[i] = self.temperature_w*random.random()*self.temperature_v[i] + self.temperature_c1*random.random()*(self.temperature_pbest[i]-self.temperature_x[i]) + self.temperature_c2*random.random()*(self.temperature_gbest-self.temperature_x[i])
            self.gas_density_v[i] = self.gas_density_w*random.random()*self.gas_density_v[i] + self.gas_density_c1*random.random()*(self.gas_density_pbest[i]-self.gas_density_x[i]) + self.gas_density_c2*random.random()*(self.gas_density_gbest-self.gas_density_x[i])
            self.lambda_x[i] = self.lambda_x[i]+self.lambda_v[i]
            self.temperature_x[i] = self.temperature_x[i]+self.temperature_v[i]
            self.gas_density_x[i] = self.gas_density_x[i]+self.gas_density_v[i]
    def getalphascore(self, lambda_est, temperature, gas_density):
        self.alpha_theo = objective_function(lambda_est, temperature, gas_density)
        score = 0
        for i in range(len(alpha_exp)):
            score = score + (alpha_exp[i] - self.alpha_theo[i]) ** 2
        return score
    def getphasescore(self):
        for i in range(self.particle_num):
            # 最小値と最大値を超えたら再び初期化
            random.seed()
            if self.lambda_x[i] < self.lambda_min or self.lambda_max < self.lambda_x[i]:
                self.lambda_x[i] = random.uniform(self.lambda_min, self.lambda_max)
            if self.temperature_x[i] < self.temperature_min or self.temperature_max < self.temperature_x[i]:
                self.temperature_x[i] = random.uniform(self.temperature_min, self.temperature_max)
            if self.gas_density_x[i] < self.gas_density_min or self.gas_density_max < self.gas_density_x[i]:
                self.gas_density_x[i] = random.uniform(self.gas_density_min, self.gas_density_max)
            self.nowscore = self.getalphascore(self.lambda_x[i], self.temperature_x[i], self.gas_density_x[i])
            if self.nowscore < self.pbestscore[i]:
                self.pbestscore[i] = self.nowscore 
                self.lambda_pbest[i] = self.lambda_x[i]
                self.temperature_pbest[i] = self.temperature_x[i]
                self.gas_density_pbest[i] = self.gas_density_x[i]
            if self.pbestscore[i] < self.gbestscore:
                self.gbestscore = self.pbestscore[i]
                self.lambda_gbest = self.lambda_pbest[i]
                self.temperature_gbest = self.temperature_pbest[i]
                self.gas_density_gbest = self.gas_density_pbest[i]
    def main(self):
            for i in range(2000):
                self.move()
                self.getphasescore()
                if i % 100 == 0 and i != 0:
                    print("{}世代目".format(i))
                    print(self.lambda_gbest, self.temperature_gbest, self.gas_density_gbest)
                    print(self.nowscore)
                if self.nowscore < border:
                    print(self.nowscore)
                    break
                elif i > 10000:
                    print("もう一度やり直してください")
                    break
            return self.alpha_theo, self.lambda_gbest, self.temperature_gbest, self.gas_density_gbest

class GA_1():
    # paremeter
    indiv_num = 200
    gene_num = 3
    all_param = np.zeros((indiv_num, gene_num))
    alpha_theo = []
    nowscore = 0
    # パラメータの最小値と最大値
    lambda_max = 696.5435
    lambda_min = 696.5425
    gas_density_max = 18
    gas_density_min = 15
    temperature_max = 500
    temperature_min = 400
    # 返り値
    alpha_theo = []
    lambda_est = 0
    temperature = 0
    gas_density = 0
    # 初期値、コンストラクタ
    def __init__(self):
        for i in range(self.indiv_num):
            random.seed()
            self.all_param[i][0] = random.uniform(self.lambda_min, self.lambda_max)
            self.all_param[i][1] = random.uniform(self.temperature_min, self.temperature_max)
            self.all_param[i][2] = random.uniform(self.gas_density_min, self.gas_density_max)

    def getalphascore(self, lambda_est, temperature, gas_density):
            self.alpha_theo = objective_function(lambda_est, temperature, gas_density)
            score = 0
            for i in range(len(alpha_exp)):
                score = score + (alpha_exp[i] - self.alpha_theo[i]) ** 2
            return score
    def next_gen(self):
        # make score list
        score_para = []
        for i in range(self.indiv_num):
            score = self.getalphascore(self.all_param[i][0], self.all_param[i][1], self.all_param[i][2])
            score_para.append(score)
        # get minimal score index
        min_indiv_index = score_para.index(min(score_para))
        self.nowscore = min(score_para)
        min_tmp = np.zeros((2, 3), dtype=float)
        min_tmp[0][0] = self.all_param[min_indiv_index][0]
        min_tmp[0][1] = self.all_param[min_indiv_index][1]
        min_tmp[0][2] = self.all_param[min_indiv_index][2]
        # get nth minimal score index
        nth = 2
        score_para_np = np.array(score_para)
        nthmin_indiv_index = np.argsort(score_para_np)[-nth]
        min_tmp[1][0] = self.all_param[nthmin_indiv_index][0]
        min_tmp[1][1] = self.all_param[nthmin_indiv_index][1]
        min_tmp[1][2] = self.all_param[nthmin_indiv_index][2]
        # パラメータの更新
        startpos_random = 0
        startpos_randamizechild = self.indiv_num-50
        startpos_child = self.indiv_num-25
        # 交叉
        # ランダムで最小値かn番目最小値から値を入手して、元のパラメータを更新
        for i in range(startpos_child, self.indiv_num):
            random.seed()
            rnd = random.randint(0, 1)
            self.all_param[i][0] = min_tmp[rnd][0]
            rnd = random.randint(0, 1)
            self.all_param[i][1] = min_tmp[rnd][1]
            rnd = random.randint(0, 1)
            self.all_param[i][2] = min_tmp[rnd][2]
        # 突然変異
        # ランダムでパラメータの値を更新
        for i in range(startpos_random, startpos_randamizechild):
            random.seed()
            self.all_param[i][0] = random.uniform(self.lambda_min, self.lambda_max)
            self.all_param[i][1] = random.uniform(self.temperature_min, self.temperature_max)
            self.all_param[i][2] = random.uniform(self.gas_density_min, self.gas_density_max)
        # 最も良かったものの値を少し変えて残す
        for i in range(startpos_randamizechild, startpos_child):
            random.seed()
            self.all_param[i][0] = min_tmp[0][0] + 0.01 * random.uniform(-1,1)
            self.all_param[i][1] = min_tmp[0][1] + 5 * random.uniform(-1,1)
            self.all_param[i][2] = min_tmp[0][2] + 1 * random.uniform(-1,1)
        for i in range(1):
            self.all_param[i][0] = min_tmp[0][0]
            self.all_param[i][1] = min_tmp[0][1]
            self.all_param[i][2] = min_tmp[0][2]
    def get_bestscore_param(self):
        score_para = []
        for i in range(self.indiv_num):
            score = self.getalphascore(self.all_param[i][0], self.all_param[i][1], self.all_param[i][2])
            score_para.append(score)
        # get minimal score index
        min_indiv_index = score_para.index(min(score_para))
        _ = self.getalphascore(self.all_param[min_indiv_index][0], self.all_param[min_indiv_index][1], self.all_param[min_indiv_index][2])
        return self.alpha_theo, self.all_param[min_indiv_index][0], self.all_param[min_indiv_index][1], self.all_param[min_indiv_index][2]
    def main(self):
        for i in range(10000):
            self.next_gen()
            if i % 50 == 0 and i != 0:
                print("{}世代目".format(i))
                print(self.nowscore)
            if self.nowscore < border:
                break
        alpha_theo, lambda_est, temperature, gas_density = self.get_bestscore_param()
        return alpha_theo, lambda_est, temperature, gas_density

class GA_2():
    # paremeter
    indiv_num = 200
    gene_num = 3
    all_param = np.zeros((indiv_num, gene_num))
    alpha_theo = []
    nowscore = 0
    # パラメータの最小値と最大値
    lambda_max = 696.5435
    lambda_min = 696.5425
    gas_density_max = 18
    gas_density_min = 14
    temperature_max = 350
    temperature_min = 250
    # 返り値
    alpha_theo = []
    lambda_est = 0
    temperature = 0
    gas_density = 0
    # 初期値、コンストラクタ
    def __init__(self):
        for i in range(self.indiv_num):
            random.seed()
            self.all_param[i][0] = random.uniform(self.lambda_min, self.lambda_max)
            self.all_param[i][1] = random.uniform(self.temperature_min, self.temperature_max)
            self.all_param[i][2] = random.uniform(self.gas_density_min, self.gas_density_max)
    def getalphascore(self, lambda_est, temperature, gas_density):
            self.alpha_theo = objective_function(lambda_est, temperature, gas_density)
            score = 0
            for i in range(len(alpha_exp)):
                score = score + (alpha_exp[i] - self.alpha_theo[i]) ** 2
            return score
    def next_gen(self):
            score_para = []
            for i in range(self.indiv_num):
                score = self.getalphascore(self.all_param[i][0], self.all_param[i][1], self.all_param[i][2])
                score_para.append(score)
            self.nowscore = min(score_para)
            score_para_np = np.array(score_para)
            score_para_np_argsort = np.argsort(score_para_np)
            score_para_min = score_para.index(min(score_para))
            min_tmp = np.zeros((2, 3), dtype=float)
            for i in range(self.indiv_num):
                # スコアが高かった順から交叉、突然変異、そのまま、が行われる。
                rnd = random.random()
                if 1-(score_para_np_argsort[i]/self.indiv_num) > rnd:
                    # kousa
                    num_tmp = random.randint(1,2)
                    if num_tmp == 1:
                        self.all_param[i][0] = self.all_param[score_para_min][0]
                    num_tmp = random.randint(1,2)
                    if num_tmp == 1:
                        self.all_param[i][1] = self.all_param[score_para_min][1]
                    num_tmp = random.randint(1,2)
                    if num_tmp == 1:
                        self.all_param[i][2] = self.all_param[score_para_min][2]
                # 突然変異
                elif (1-(score_para_np_argsort[i]/self.indiv_num))+(score_para_np_argsort[i]/self.indiv_num)*(1/2) > rnd:
                    if random.random() < (1/3):
                        self.all_param[i][0] = random.uniform(self.lambda_min, self.lambda_max)
                    elif random.random() < (2/3):
                        self.all_param[i][1] = random.uniform(self.temperature_min, self.temperature_max)
                    elif random.random() < (3/3):
                        self.all_param[i][1] = random.uniform(self.gas_density_min, self.gas_density_max)
                else:
                    # そのまま残す
                    pass
    def get_bestscore_param(self):
        score_para = []
        for i in range(self.indiv_num):
            score = self.getalphascore(self.all_param[i][0], self.all_param[i][1], self.all_param[i][2])
            score_para.append(score)
        # get minimal score index
        min_indiv_index = score_para.index(min(score_para))
        _ = self.getalphascore(self.all_param[min_indiv_index][0], self.all_param[min_indiv_index][1], self.all_param[min_indiv_index][2])
        return self.alpha_theo, self.all_param[min_indiv_index][0], self.all_param[min_indiv_index][1], self.all_param[min_indiv_index][2]
    def main(self):
        for i in range(10000):
            self.next_gen()
            if i % 100 == 0 and i != 0:
                print("{}世代目".format(i))
                print(self.nowscore)
            if self.nowscore < border:
                break
        alpha_theo, lambda_est, temperature, gas_density = self.get_bestscore_param()
        return alpha_theo, lambda_est, temperature, gas_density

# ホタルアルゴリズム
class HOTARU():
    # paremeter
    firefly_num = 40
    fireflys = np.zeros((firefly_num, 3))
    attracting_degree = 1.0
    absorb = 0.5
    alpha = 0.5
    nowscore = 100000
    # パラメータの最小値と最大値
    lambda_max = 696.5435
    lambda_min = 696.5425
    temperature_max = 500
    temperature_min = 400
    gas_density_max = 18
    gas_density_min = 15
    # 返り値
    alpha_theo = []
    lambda_est = 0
    temperature = 0
    gas_density = 0
    def __init__(self):
        for i in range(self.firefly_num):
            self.fireflys[i][0] = random.uniform(self.lambda_min, self.lambda_max)
            self.fireflys[i][1] = random.uniform(self.temperature_min, self.temperature_max)
            self.fireflys[i][2] = random.uniform(self.gas_density_min, self.gas_density_max)
    def getalphascore(self, lambda_est, temperature, gas_density):
        self.alpha_theo = objective_function(lambda_est, temperature, gas_density)
        score = 0
        for i in range(len(alpha_exp)):
            score = score + (alpha_exp[i] - self.alpha_theo[i]) ** 2
        return score
    def step(self):
        max_pos = np.array([
            [self.lambda_max, self.temperature_max, self.gas_density_max], 
            [self.lambda_min, self.temperature_min, self.gas_density_min]
            ], dtype=float)
        for i in range(self.firefly_num):
            for j in range(self.firefly_num):
                if i == j:
                    continue
                score_i = self.getalphascore(self.fireflys[i][0], self.fireflys[i][1], self.fireflys[i][2])
                score_j = self.getalphascore(self.fireflys[j][0], self.fireflys[j][1], self.fireflys[j][2])
                if score_i > score_j:
                    pos_i = self.fireflys[i]
                    pos_j = self.fireflys[j]
                    # ユークリッド距離
                    d = np.linalg.norm(pos_j - pos_i)
                    # 正規化
                    d /= np.linalg.norm(max_pos)
                    # 誘引度
                    attract = self.attracting_degree * (math.exp(-self.absorb * (d ** 2)))
                    r_range_lambda = 0.0001
                    r_range_temperature = 30.0
                    r_range_gas= 0.1
                    pos_i[0] = pos_i[0] + attract * (pos_j[0] - pos_i[0]) + self.alpha * random.uniform(-r_range_lambda, r_range_lambda)
                    pos_i[1] = pos_i[1] + attract * (pos_j[1] - pos_i[1]) + self.alpha * random.uniform(-r_range_temperature, r_range_temperature)
                    pos_i[2] = pos_i[2] + attract * (pos_j[2] - pos_i[2]) + self.alpha * random.uniform(-r_range_gas, r_range_gas)
                    self.fireflys[i] = pos_i
                    #print(self.fireflys[i])
                    if self.nowscore > self.getalphascore(self.fireflys[i][0], self.fireflys[i][1], self.fireflys[i][2]):
                        self.nowscore = self.getalphascore(self.fireflys[i][0], self.fireflys[i][1], self.fireflys[i][2])
                        self.lambda_est = self.fireflys[i][0]
                        self.temperature = self.fireflys[i][1]
                        self.gas_density = self.fireflys[i][2]
    def main(self):
        for i in range(100):
            self.step()
            # print(self.lambda_est, self.temperature, self.gas_density)
            if i % 1 == 0 and i != 0:
                print("{}世代目".format(i))
                print(self.nowscore)
            if self.nowscore < border:
                break
        return self.alpha_theo, self.lambda_est, self.temperature, self.gas_density

# コウモリアルゴリズム
class BAT():
    # paremeter
    bat_num = 50
    bats = np.zeros((bat_num, 3), dtype=float)
    bats_v = np.zeros((bat_num, 3), dtype=float)
    bats_pulse = np.zeros(bat_num, dtype=float)
    bats_vol = np.ones(bat_num, dtype=float)
    good_bat_num = int(bat_num/10)
    freq_max = 1.0
    freq_min = 0.0
    volume_update_rate = 0.9
    pulse_convergence_value = 0.7
    pulse_convergence_speed = 0.2
    step_cnt = 0
    nowscore = 100000
    # パラメータの最小値と最大値
    lambda_max = 696.5435
    lambda_min = 696.5425
    temperature_max = 500
    temperature_min = 400
    gas_density_max = 17
    gas_density_min = 15
    # 返り値
    alpha_theo = []
    lambda_est = 0
    temperature = 0
    gas_density = 0
    def __init__(self):
        for i in range(self.bat_num):
            self.bats[i][0] = random.uniform(self.lambda_min, self.lambda_max)
            self.bats[i][1] = random.uniform(self.temperature_min, self.temperature_max)
            self.bats[i][2] = random.uniform(self.gas_density_min, self.gas_density_max)
    def getalphascore(self, lambda_est, temperature, gas_density):
        self.alpha_theo = objective_function(lambda_est, temperature, gas_density)
        score = 0
        for i in range(len(alpha_exp)):
            score = score + (alpha_exp[i] - self.alpha_theo[i]) ** 2
        return score
    def reset(self):
        for i in range(self.bat_num):
            # 最小値と最大値を超えたら再び初期化
            random.seed()
            if self.bats[i][0] < self.lambda_min or self.lambda_max < self.bats[i][0]:
                self.bats[i][0] = random.uniform(self.lambda_min, self.lambda_max)
            if self.bats[i][1] < self.temperature_min or self.temperature_max < self.bats[i][1]:
                self.bats[i][1] = random.uniform(self.temperature_min, self.temperature_max)
            if self.bats[i][2] < self.gas_density_min or self.gas_density_max < self.bats[i][2]:
                self.bats[i][2] = random.uniform(self.gas_density_min, self.gas_density_max)
    def sort_bats(self):
        score = np.zeros(self.bat_num)
        for i in range(self.bat_num):
            score[i] = self.getalphascore(self.bats[i][0], self.bats[i][1], self.bats[i][2])
        sorted_idx = np.argsort(score)
        score = score[sorted_idx]
        self.bats = self.bats[sorted_idx]
        self.bats_pulse = self.bats_pulse[sorted_idx]
        self.bats_vol = self.bats_vol[sorted_idx]
    def step(self):
        # 配列をソート
        self.sort_bats()
        # 最良コウモリに近づく
        pos_best = self.bats[0] 
        for i in range(self.bat_num):
            pos = self.bats[i]
            # 周波数の計算
            freq = random.uniform(self.freq_min, self.freq_max)
            # 速度の計算
            self.bats_v[i] = self.bats_v[i] * freq * (pos_best - pos)
            # 位置を更新
            self.bats[i] = self.bats[i] + self.bats_v[i]
            score_bati = self.getalphascore(self.bats[i][0], self.bats[i][1], self.bats[i][2])
        # 良いコウモリの近傍に移動
            score_newbat1 = 100000
            if self.bats_pulse[i] > random.random():
                r = random.randint(0, self.good_bat_num)
                pos_good = self.bats[r]
                vol_ave = np.average(self.bats_vol)
                new_bat1 = 3 * [0]
                new_bat1[0] = pos_good[0] + vol_ave * random.uniform(-0.0001, 0.0001)
                new_bat1[1] = pos_good[1] + vol_ave * random.uniform(-1.5, 1.5)
                new_bat1[2] = pos_good[2] + vol_ave * random.uniform(-0.1, 0.1)
                #print(new_bat1)
                score_newbat1 = self.getalphascore(new_bat1[0], new_bat1[1], new_bat1[2])
            # ランダムに生成
            new_bat2 = 3 * [0.0]
            new_bat2[0] = random.uniform(self.lambda_min, self.lambda_max)
            new_bat2[1] = random.uniform(self.temperature_min, self.temperature_max)
            new_bat2[2] = random.uniform(self.gas_density_min, self.gas_density_max)
            score_newbat2 = self.getalphascore(new_bat2[0], new_bat2[1], new_bat2[2])
            # 新しい位置が元の位置より評価が高いかどうか
            if score_newbat1 < score_bati or score_newbat2 < score_bati: 
                if random.uniform(-1, 1) < self.bats_vol[i]:
                    # 新しい位置に変更と音量とパルス率の更新
                    if score_newbat1 >= score_newbat2:
                        self.bats[i] = new_bat2
                    else:
                        self.bats[i] = new_bat1
                    # パルス率の更新
                    self.bats_pulse[i] = self.pulse_convergence_value * (1-math.exp(-self.pulse_convergence_speed))
                    # 音量の更新
                    self.bats_vol[i] = self.volume_update_rate * self.bats_vol[i]
                else:
                    pass
            else:
                pass
            self.step_cnt += 1
            if self.nowscore > self.getalphascore(self.bats[i][0], self.bats[i][1], self.bats[i][2]):
                self.nowscore = self.getalphascore(self.bats[i][0], self.bats[i][1], self.bats[i][2])
                self.lambda_est = self.bats[i][0]
                self.temperature = self.bats[i][1]
                self.gas_density = self.bats[i][2]
            #self.reset()
    def main(self):
        for i in range(500):
            self.step()
            # print(self.lambda_est, self.temperature, self.gas_density)
            if i % 20 == 0 and i != 0:
                print("{}世代目".format(i))
                print(self.nowscore)
            if self.nowscore < border:
                break
        return self.alpha_theo, self.lambda_est, self.temperature, self.gas_density

class CUCKOO():
    # paremeter
    nest_num = 10
    nests = np.zeros((nest_num, 3), dtype=float)
    good_nest_num = int(nest_num/10)
    epsilon = 0.1
    badnest_num = int(nest_num/1.5)
    nowscore = 100000
    # パラメータの最小値と最大値
    lambda_max = 696.5435
    lambda_min = 696.5425
    temperature_max = 500
    temperature_min = 400
    gas_density_max = 17
    gas_density_min = 15
    # 返り値
    alpha_theo = []
    lambda_est = 0
    temperature = 0
    gas_density = 0
    def __init__(self):
        for i in range(self.nest_num):
            self.nests[i][0] = random.uniform(self.lambda_min, self.lambda_max)
            self.nests[i][1] = random.uniform(self.temperature_min, self.temperature_max)
            self.nests[i][2] = random.uniform(self.gas_density_min, self.gas_density_max)
    def getalphascore(self, lambda_est, temperature, gas_density):
        self.alpha_theo = objective_function(lambda_est, temperature, gas_density)
        score = 0
        for i in range(len(alpha_exp)):
            score = score + (alpha_exp[i] - self.alpha_theo[i]) ** 2
        return score
    def reset(self):
        for i in range(self.nest_num):
            # 最小値と最大値を超えたら再び初期化
            random.seed()
            if self.nests[i][0] < self.lambda_min or self.lambda_max < self.nests[i][0]:
                self.nests[i][0] = random.uniform(self.lambda_min, self.lambda_max)
            if self.nests[i][1] < self.temperature_min or self.temperature_max < self.nests[i][1]:
                self.nests[i][1] = random.uniform(self.temperature_min, self.temperature_max)
            if self.nests[i][2] < self.gas_density_min or self.gas_density_max < self.nests[i][2]:
                self.nests[i][2] = random.uniform(self.gas_density_min, self.gas_density_max)
    def sort_nests(self):
        random.seed()
        score = np.zeros(self.nest_num)
        for i in range(self.nest_num):
            score[i] = self.getalphascore(self.nests[i][0], self.nests[i][1], self.nests[i][2])
        sorted_idx = np.argsort(score)
        score = score[sorted_idx]
        self.nests = self.nests[sorted_idx]
    def step(self):
        r = random.randint(0, self.nest_num-1)
        new_nest = self.nests[r]
        if random.random() < self.epsilon:
            new_nest[0] = random.uniform(self.lambda_min, self.lambda_max)
            new_nest[1] = random.uniform(self.temperature_min, self.temperature_max)
            new_nest[2] = random.uniform(self.gas_density_min, self.gas_density_max)
        r = random.randint(0, self.nest_num-1)
        score_r = self.getalphascore(self.nests[r][0], self.nests[r][1], self.nests[r][2])
        score_new = self.getalphascore(new_nest[0], new_nest[1], new_nest[2])
        if score_r > score_new:
            self.nests[r][0] = new_nest[0]
            self.nests[r][1] = new_nest[1]
            self.nests[r][2] = new_nest[2]
        self.sort_nests()
        for i in range(self.nest_num - self.badnest_num, self.nest_num):
            self.nests[i][0] = random.uniform(self.lambda_min, self.lambda_max)
            self.nests[i][1] = random.uniform(self.temperature_min, self.temperature_max)
            self.nests[i][2] = random.uniform(self.gas_density_min, self.gas_density_max)
        for i in range(self.nest_num):
            if self.nowscore > self.getalphascore(self.nests[i][0], self.nests[i][1], self.nests[i][2]):
                self.nowscore = self.getalphascore(self.nests[i][0], self.nests[i][1], self.nests[i][2])
                self.lambda_est = self.nests[i][0]
                self.temperature = self.nests[i][1]
                self.gas_density = self.nests[i][2]
            #self.reset()
    def main(self):
        for i in range(8000):
            self.step()
            # print(self.lambda_est, self.temperature, self.gas_density)
            if i % 500 == 0 and i != 0:
                print("{}世代目".format(i))
                print(self.nowscore)
            if self.nowscore < border:
                break
        return self.alpha_theo, self.lambda_est, self.temperature, self.gas_density


class BEE():
    # paremeter
    bee_num = 50
    flowers = np.zeros((bee_num, 3), dtype=float)
    flower_count = np.zeros(bee_num, dtype=float)
    follow_bee = 10
    visit_max = 10
    nowscore = 100000
    # パラメータの最小値と最大値
    lambda_max = 696.5435
    lambda_min = 696.5425
    temperature_max = 500
    temperature_min = 450
    gas_density_max = 17
    gas_density_min = 16
    # 返り値
    alpha_theo = []
    lambda_est = 0
    temperature = 0
    gas_density = 0
    def __init__(self):
        for i in range(self.bee_num):
            self.flowers[i][0] = random.uniform(self.lambda_min, self.lambda_max)
            self.flowers[i][1] = random.uniform(self.temperature_min, self.temperature_max)
            self.flowers[i][2] = random.uniform(self.gas_density_min, self.gas_density_max)
    def getalphascore(self, lambda_est, temperature, gas_density):
        self.alpha_theo = objective_function(lambda_est, temperature, gas_density)
        score = 0
        for i in range(len(alpha_exp)):
            score = score + (alpha_exp[i] - self.alpha_theo[i]) ** 2
        return score
    def reset(self):
        for i in range(self.bee_num):
            # 最小値と最大値を超えたら再び初期化
            random.seed()
            if self.flowers[i][0] < self.lambda_min or self.lambda_max < self.flowers[i][0]:
                self.flowers[i][0] = random.uniform(self.lambda_min, self.lambda_max)
            if self.flowers[i][1] < self.temperature_min or self.temperature_max < self.flowers[i][1]:
                self.flowers[i][1] = random.uniform(self.temperature_min, self.temperature_max)
            if self.flowers[i][2] < self.gas_density_min or self.gas_density_max < self.flowers[i][2]:
                self.flowers[i][2] = random.uniform(self.gas_density_min, self.gas_density_max)
    def reset_new(self, new_flower):
        # 最小値と最大値を超えたら再び初期化
        random.seed()
        if new_flower[0] < self.lambda_min or self.lambda_max < new_flower[0]:
            new_flower[0] = random.uniform(self.lambda_min, self.lambda_max)
        if new_flower[1] < self.temperature_min or self.temperature_max < new_flower[1]:
            new_flower[1] = random.uniform(self.temperature_min, self.temperature_max)
        if new_flower[2] < self.gas_density_min or self.gas_density_max < new_flower[2]:
            new_flower[2] = random.uniform(self.gas_density_min, self.gas_density_max)

    def selectFlower(self):
        weights = [self.getalphascore(x[0], x[1], x[2]) for x in self.flowers]
        r = random.random() * sum(weights)
        num = 0
        for i, weights in enumerate(weights):
            num += weights
            if r <= num:
                return self.flowers[i], i
    def step(self):
        # 収穫バチフェーズ
        for i in range(self.bee_num):
            new_flower = self.flowers[i]
            k = random.randint(0, len(new_flower)-1)
            rnd = random.randint(0, self.bee_num-1)
            bee2 = self.flowers[rnd]
            if k == 0:
                new_flower[k] = new_flower[k] + 0.05*(random.random()*2-1)*(new_flower[k]-bee2[k])
            elif k == 1:
                new_flower[k] = new_flower[k] + 0.05*(random.random()*2-1)*(new_flower[k]-bee2[k])
            else:
                new_flower[k] = new_flower[k] + 0.01*(random.random()*2-1)*(new_flower[k]-bee2[k])
            self.reset_new(new_flower)
            score_new_flower = self.getalphascore(new_flower[0], new_flower[1], new_flower[2])
            score_now_bee = self.getalphascore(self.flowers[i][0], self.flowers[i][1], self.flowers[i][2])
            if score_new_flower < score_now_bee:
                self.flowers[i] = new_flower
                if score_new_flower < self.nowscore:
                    self.nowscore = score_new_flower
                    self.lambda_est = new_flower[0]
                    self.temperature = new_flower[1]
                    self.gas_density = new_flower[2]
            self.flower_count[i] += 1
        # 追従バチフェーズ
        for i in range(self.follow_bee):
            new_flower, j = self.selectFlower()
            #self.flowers[j] = new_flower
            self.flower_count[j] += 1
            new_flower_score = self.getalphascore(new_flower[0], new_flower[1], new_flower[2])
            if new_flower_score < self.nowscore:
                self.nowscore = new_flower_score
                self.lambda_est = new_flower[0]
                self.temperature = new_flower[1]
                self.gas_density = new_flower[2]
        # 偵察バチフェーズ
        for i in range(len(self.flowers)):
            if self.visit_max < self.flower_count[i]:
                new_flower = np.zeros(3, dtype=float)
                new_flower[0] = random.uniform(self.lambda_min, self.lambda_max)
                new_flower[1] = random.uniform(self.temperature_min, self.temperature_max)
                new_flower[2] = random.uniform(self.gas_density_min, self.gas_density_max)
                self.flower_count[i] = 0
                new_flower_score = self.getalphascore(new_flower[0], new_flower[1], new_flower[2])
                if new_flower_score < self.nowscore:
                    self.nowscore = new_flower_score
                    self.lambda_est = new_flower[0]
                    self.temperature = new_flower[1]
                    self.gas_density = new_flower[2]
            
    def main(self):
        for i in range(1000):
            self.step()
            # print(self.lambda_est, self.temperature, self.gas_density)
            if i % 10 == 0 and i != 0:
                print("{}世代目".format(i))
                print(self.nowscore)
            if self.nowscore < border:
                break
        return self.alpha_theo, self.lambda_est, self.temperature, self.gas_density


# メイン関数、これが実行される
if __name__ == "__main__":
    # 測定機器のサンプリングレート
    sampling_rate = 2000
    # 最適化アルゴリズムの理論値と実測値の最小二乗のしきい値, これがカーブフィッティングの精度になる、実験データ毎に変更
    #border = 0.191
    border = 0.715
    # FPI信号間隔
    FPI_signal_interval = 101
    # ファイルのパスの設定
    path_dir = '/home/rune/desktop/zikken/curvefitting/'
    path_LonPon = path_dir + 'LonPon.csv'
    path_LonPoff = path_dir + 'LonPoff.csv'
    path_LoffPon = path_dir + 'LoffPon.csv'
    path_LoffPoff = path_dir + 'LoffPoff.csv'
    # データの読み込みと整形
    LonPon = read_csv(path_LonPon)
    LonPoff = read_csv(path_LonPoff)
    LoffPon = read_csv(path_LoffPon)
    LoffPoff = read_csv(path_LoffPoff)
    # 整形したデータの出力(元のVBAでデータを処理したい場合)
    #export_shaped_csv()
    # αの実験値を求める =1-((LonPon-LoffPon)/(LonPoff-LoffPoff))
    # 求める配列(平均を取る?一部分を切り取る?)を作って、そこから求める
    # nは何番目の半周期かを設定、奇数なら下降、偶数なら上昇を表す
    n = 1
    I1 = LonPon[['ch0']].loc[(sampling_rate/2)*(n-1):(sampling_rate/2)*n, :]
    I3 = LoffPon[['ch0']].loc[(sampling_rate/2)*(n-1):(sampling_rate/2)*n, :]
    I2 = LonPoff[['ch0']].loc[(sampling_rate/2)*(n-1):(sampling_rate/2)*n, :]
    I4 = LoffPoff[['ch0']].loc[(sampling_rate/2)*(n-1):(sampling_rate/2)*n, :]
    # 実験値αの配列
    alpha_exp = []
    for i in range(int((sampling_rate/2)*(n-1)), int((sampling_rate/2)*n)):
        alpha_i = 1-((I1.at[i, 'ch0']-I3.at[i, 'ch0'])/(I2.at[i, 'ch0']-I4.at[i, 'ch0']))
        alpha_exp.append(alpha_i)
    # 実験値αの最大値
    max_value = max(alpha_exp)
    # 実験値αの最大セル位置
    max_index = alpha_exp.index(max(alpha_exp))
    # 実験値αの中から吸収部分だけ取り出す
    slice_range = 200
    #alpha_exp = alpha_exp[max_index-slice_range:max_index+slice_range]
    # 実験値αの配列数
    index = list(range(0, len(alpha_exp)))


    # αの理論値を求める、フィッティングを行って、温度と密度を出す
    # 定数
    lambda_0 = 696.543*(10**(-9))
    C = 299800000
    R = 8314.462618
    Ar_M = 39.958
    optical_path_length = 0.3
    tmp_center_pos = 696.543051064163
    etaron_width = 1.5 * (10**9)
    # 定数1=光速*遷移波長*√Arの原子量/2*気体定数 = 10.236449341204224
    const_1 = C * lambda_0 * math.sqrt(Ar_M/(2*R))
    # 定数2=(𝜆0^3 𝑔_2 𝐴_21 𝑙)/(8𝜋𝑔_1 ) √(𝑀/2𝜋𝑅)=4.2773239588554036e-16
    const_2 = (((lambda_0**3)*3*6390000*optical_path_length)/(8*math.pi*5))*math.sqrt(Ar_M/(2*math.pi*R))
    # λ0でのFSL=0.0024274790736274186
    FSL_ramda0 = (lambda_0 ** 2) / C * etaron_width * (10 ** 9)
    # データ間隔=2.45199906427012e-05
    delta_lambda = FSL_ramda0 / (FPI_signal_interval - 1)
    # 半値
    # sorted_alpha_exp = sorted(alpha_exp)
    # half_value = sorted_alpha_exp[int(len(alpha_exp)/2)]
    # 前半位置
    # front_position = alpha_exp.index(half_value)
    # 後半位置
    # back_position = front_position + max_index
    # estimated temperature
    # doppler_width = ( back_position - front_position )/FPI_signal_interval * etaron_width
    # transition = C / lambda_0
    # temperature_est = optical_path_length * C * C / (8 * R * math.log(2)) * ( (doppler_width/transition) ** 2)
    # 実験値積分
    # exp_integration = sum(alpha_exp) + FSL_ramda0
    # 実験値λの取得
    lambda_theo = get_lambda_theo()
    #alpha_exp = np.array(alpha_exp)
    #lambda_theo = np.array(lambda_theo)

    # 最適化
    #opt_formula = BAT()
    #opt_formula = HOTARU()
    opt_formula = PSO()
    #opt_formula = CUCKOO()
    #opt_formula = BEE()
    #opt_formula = GA_1()
    #opt_formula = GA_2()
    alpha_theo, lambda_est, temperature, gas_density = opt_formula.main()
    
    df_alpha = pd.DataFrame(
        data = {
            '実験値α':alpha_exp, 
            '理論値α':alpha_theo
            }
    )
    #df_alpha.to_csv(path_dir + 'new_alpha.csv',  encoding = 'utf-8')

    print("gas_temperature[K] : {}".format(temperature))
    print("gas_density[/m^3] : {}".format(10**gas_density))
    print("gas_density(N) : {}".format(gas_density))
    # plot
    plot_graph()
