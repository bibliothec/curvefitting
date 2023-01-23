import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from numba import jit

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

def get_lambda_theo():
    # 理論値のλの配列(x軸の値になる)
    lambda_theo = []
    for i in index:
        lambda_i = tmp_center_pos + delta_lambda * (i - max_index)
        lambda_theo.append(lambda_i)
    return lambda_theo

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
    particle_num = 5
    unknown_num = 3
    alpha_theo = []
    nowscore = 10000000
    # 重み
    lambda_w = 0.4
    lambda_c1 = 1.8 
    lambda_c2 = 2.1
    temperature_w = 0.7
    temperature_c1 = 1.8
    temperature_c2 = 2.1
    gas_density_w = 0.7
    gas_density_c1 = 1.8
    gas_density_c2 = 2.1
    # パラメータの最小値と最大値
    lambda_max = 696.5435
    lambda_min = 696.5425
    gas_density_max = 20
    gas_density_min = 10
    temperature_max = 600
    temperature_min = 300
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
        for i in range(self.particle_num-1):
            # 位置と速度の初期化
            # self.lambda_x[i] = lambda_0*10**9 + (self.lambda_max + self.lambda_min) * 0.1 * random.random()
            self.lambda_x[i] = 696.543
            # self.temperature_x[i] = exp_integration + 0.8 * (random.random() - 0.5)
            self.temperature_x[i] = random.uniform(self.temperature_min, self.temperature_max) # round(self.temperature_min + ((self.temperature_max - self.temperature_min) * random.random()), 3)
            # self.gas_density_x[i] = temperature_est + (random.random() - 0.5 ) * 200
            self.gas_density_x[i] = random.uniform(self.gas_density_min, self.gas_density_min) # round(self.gas_density_min + ((self.gas_density_max - self.gas_density_min) * random.random()), 3)
            self.lambda_pbest[i] = self.lambda_x[i]
            self.temperature_pbest[i] = self.temperature_x[i]
            self.gas_density_pbest[i] = self.gas_density_x[i]
            v_keisu_tmp = 0.4
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
        for i in range(self.particle_num-1):
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
        for i in range(self.particle_num-1):
            # 最小値と最大値を超えたら再び初期化
            random.seed()
            if self.temperature_x[i] < self.temperature_min or self.temperature_max < self.temperature_x[i]:
                self.temperature_x[i] = random.uniform(self.temperature_min, self.temperature_max)
            if self.gas_density_x[i] < self.gas_density_min or self.gas_density_max < self.gas_density_x[i]:
                self.gas_density_x[i] = random.uniform(self.gas_density_min, self.gas_density_max)
            if self.lambda_x[i] < self.lambda_min or self.lambda_max < self.lambda_x[i]:
                self.lambda_x[i] = random.uniform(self.lambda_min, self.lambda_max)
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
            for i in range(10000):
                self.move()
                self.getphasescore()
                # print(self.nowscore)
                if i % 1000 == 0 and i != 0:
                    print(f"{i}世代目")
                if self.nowscore < border:
                    break
                elif i > 1000:
                    print("もう一度やり直してください")
                    break
            return self.alpha_theo, self.lambda_gbest, self.temperature_gbest, self.gas_density_gbest

class GA():
    # paremeter
    indiv_num = 200
    gene_num = 3
    all_param = np.zeros((indiv_num, gene_num))
    alpha_theo = []
    nowscore = 0
    # パラメータの最小値と最大値
    lambda_max = 696.5435
    lambda_min = 696.5425
    gas_density_max = 19
    gas_density_min = 12
    temperature_max = 500
    temperature_min = 300

    # 初期値、コンストラクタ
    def __init__(self) -> None:
        for i in range(self.indiv_num):
            random.seed()
            self.all_param[i][0] = 696.543
            # self.all_param[i][0] = self.lambda_min + ((self.lambda_max - self.lambda_min) * random.random())
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
        # nthmin_indiv_index = np.where(score_para_np==np.sort(score_para_np)[-nth])
        nthmin_indiv_index = np.argsort(score_para_np)[-nth]
        min_tmp[1][0] = self.all_param[nthmin_indiv_index][0]
        min_tmp[1][1] = self.all_param[nthmin_indiv_index][1]
        min_tmp[1][2] = self.all_param[nthmin_indiv_index][2]
        # パラメータの更新
        startpos_random = 0
        startpos_randamizechild = self.indiv_num-15
        startpos_child = self.indiv_num-6
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
            self.all_param[i][0] = min_tmp[0][0] + 0.1 * random.random()
            self.all_param[i][1] = min_tmp[0][1] + 10 * random.random()
            self.all_param[i][2] = min_tmp[0][2] + 5 * random.random()

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
                print(f"{i}世代目")
            if self.nowscore < border:
                break
        alpha_theo, lambda_est, temperature, gas_density = self.get_bestscore_param()
        return alpha_theo, lambda_est, temperature, gas_density



# メイン関数、これが実行される
if __name__ == "__main__":
    # 測定機器のサンプリングレート
    sampling_rate = 2000
    # 最適化アルゴリズムの理論値と実測値の最小二乗のしきい値, これがカーブフィッティングの精度になる、0.45程度より低くはならない
    border = 0.48
    # FPI信号間隔
    FPI_signal_interval = 100
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
    export_shaped_csv()
    # αの実験値を求める =1-((LonPon-LoffPon)/(LonPoff-LoffPoff))
    # 求める配列(平均を取る?一部分を切り取る?)を作って、そこから求める
    # nは何番目の半周期かを設定、奇数なら下降、偶数なら上昇を表す
    n = 2
    I1 = LonPon[['ch0']].loc[(sampling_rate/2)*(n-1):(sampling_rate/2)*n, :]
    I3 = LoffPon[['ch0']].loc[(sampling_rate/2)*(n-1):(sampling_rate/2)*n, :]
    I2 = LonPoff[['ch0']].loc[(sampling_rate/2)*(n-1):(sampling_rate/2)*n, :]
    I4 = LoffPoff[['ch0']].loc[(sampling_rate/2)*(n-1):(sampling_rate/2)*n, :]
    # 実験値αの配列
    alpha_exp = []
    for i in range(int((sampling_rate/2)*(n-1)), int((sampling_rate/2)*n)):
        alpha_i = 1-((I1.at[i, 'ch0']-I3.at[i, 'ch0'])/(I2.at[i, 'ch0']-I4.at[i, 'ch0']))
        alpha_exp.append(alpha_i)
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
    # 実験値αの最大値
    max_value = max(alpha_exp)
    # 実験値αの最大セル位置
    max_index = alpha_exp.index(max(alpha_exp))
    # 実験値αの配列数
    index = list(range(0, len(alpha_exp)))
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

    # 最適化
    opt_formula = PSO()
    alpha_theo, lambda_est, temperature, gas_density = opt_formula.main()
    
    df_alpha = pd.DataFrame(
        data = {
            '実験値α':alpha_exp, 
            '理論値α':alpha_theo
            }
    )
    df_alpha.to_csv(path_dir + 'new_alpha.csv',  encoding = 'utf-8')

    print(f"gas_temperature : {temperature}")
    print(f"gas_density : {10**gas_density}")
    # plot
    plot_graph()
