import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import numpy as np
from numba import jit

def read_csv(path):
    # ch0は光電子増倍管, ch1はピエゾ, ch2はエタロン
    df = pd.read_csv(path, sep=',', header=None,  names=['SampleNum', 'DataTime', 'ch0', 'ch1', 'ch2', 'Events'])
    # CSV上の余計なデータの削除
    # 上の複数行
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
    # print(df)
    return df

def export_shaped_csv():
    LonPon.to_csv(path_dir + 'new_LonPon.csv',  encoding = 'utf-8')
    LoffPon.to_csv(path_dir + 'new_LoffPon.csv',  encoding = 'utf-8')
    LonPoff.to_csv(path_dir + 'new_LonPoff.csv',  encoding = 'utf-8')
    LoffPoff.to_csv(path_dir + 'new_LoffPoff.csv',  encoding = 'utf-8')


def objective_function(alpha_exp, temperature=400, gas_density=5.46*(10**17)):
    # 定数
    lambda_0 = 696.543*(10**(-9))
    C = 299800000
    R = 8314.462618
    Ar_M = 39.958
    tmp_center_pos = 696.543051064163
    doppler_width = 1.5 * (10**9)
    # 定数1=光速*遷移波長*√Arの原子量/2*気体定数
    const_1 = C * lambda_0 * math.sqrt(Ar_M/(2*R))
    # 定数2=(𝜆0^3 𝑔_2 𝐴_21 𝑙)/(8𝜋𝑔_1 ) √(𝑀/2𝜋𝑅)
    const_2 = (((lambda_0**3)*3*6390000*0.3)/(8*math.pi*5))*math.sqrt(Ar_M/(2*math.pi*R))
    # λ0でのFSL
    FSL_ramda0 = (lambda_0 ** 2) / C * doppler_width  * (10 ** 9)
    # FPI信号間隔
    FPI_signal_interval = 100
    # データ間隔
    delta_lambda = FSL_ramda0 / (FPI_signal_interval - 1)
    # 実験値αの最大セル位置
    max_index = alpha_exp.index(max(alpha_exp))
    # 実験値αの配列数
    index = list(range(0, len(alpha_exp)))
    # 理論値のλの配列(x軸の値になる)
    lambda_theo = []
    # 理論値αの配列
    alpha_theo = []
    for i in index:
        lambda_i = tmp_center_pos + delta_lambda * (i - max_index)
        lambda_theo.append(lambda_i)
        # print(lambda_i)
        S1 = (1/(lambda_i*(10**(-9))))-(1/lambda_0)
        S2 = const_1/math.sqrt(temperature)
        S = math.e ** (- (S1 * S2) ** 2)
        # alpha_i = 1 - math.e ** (-(const_2 / (math.sqrt(temperature) * gas_density * S)))
        alpha_i = 1 - math.e ** (-(const_2 * (10 ** (gas_density)) / (math.sqrt(temperature) * gas_density * S)))
        # alpha_i = 1-math.e**((-1*const_2*(10**gas_density)/math.sqrt(temperature))*math.e**(-1*((const_1/math.sqrt(temperature)*((1/((lambda_0+((i-max_index)*delta_lambda))*(10**(-9)))))-(1/(lambda_0*(10**(-9))))))**2))
        alpha_theo.append(alpha_i)
        print(alpha_i)
    return lambda_theo, alpha_theo, temperature, gas_density

def plot_graph():
    plt.figure(figsize = (10,6), facecolor='lightblue')
    plt.plot(lambda_theo, alpha_exp, color='blue', label='α(exp)')
    plt.plot(lambda_theo, alpha_theo, color='green', label='α(cal)')
    plt.legend(loc = 'upper right')
    plt.show()

class PSO():
    # 各粒子の位置更新
    def update_positions(self, positions, velocities):
        positions += velocities
        return positions
    # 各粒子の速度更新
    def update_velocities(self, positions, velocities, personal_best_positions, global_best_particle_position, w=0.5, ro_max=0.14):
        rc1 = random.uniform(0, ro_max)
        rc2 = random.uniform(0, ro_max)
        velocities = velocities * w + rc1 * (personal_best_positions - positions) + rc2 * (global_best_particle_position - positions)
        return velocities
    def main(self):
        number_of_particles = 10
        dimensions = 2
        limit_times = 1000
        xy_min, xy_max = 0, 1000
        # 各粒子の位置
        positions = np.array([[random.uniform(xy_min, xy_max) for _ in range(dimensions)] for _ in range(number_of_particles)])
        # 各粒子の速度
        velocities = np.zeros(positions.shape)
        # 各粒子ごとのパーソナルベスト位置
        personal_best_positions = np.copy(positions)
        # 各粒子ごとのパーソナルベストの値
        personal_best_scores = np.apply_along_axis(objective_function, 1, personal_best_positions)
        # グローバルベストの粒子ID
        global_best_particle_id = np.argmin(personal_best_scores)
        # グローバルベスト位置
        global_best_particle_position = personal_best_positions[global_best_particle_id]
        # 規定回数
        for T in range(limit_times):
            # 速度更新
            velocities = update_velocities(positions, velocities, personal_best_positions, global_best_particle_position)
            # 位置更新
            positions = update_positions(positions, velocities)
            # パーソナルベストの更新
            for i in range(number_of_particles):
                score = objective_function(positions[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
            # グローバルベストの更新
            global_best_particle_id = np.argmin(personal_best_scores)
            global_best_particle_position = personal_best_positions[global_best_particle_id]


if __name__ == "__main__":
    # 測定機器のサンプリングレート
    sampling_rate = 2000
    # ファイルのパスの設定
    path_dir = '/home/rune/desktop/zikken/カーブフィッティング/'
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
    n = 1
    I1 = LonPon[['ch0']].loc[(sampling_rate/2)*(n-1):(sampling_rate/2)*n, :]
    I3 = LoffPon[['ch0']].loc[(sampling_rate/2)*(n-1):(sampling_rate/2)*n, :]
    I2 = LonPoff[['ch0']].loc[(sampling_rate/2)*(n-1):(sampling_rate/2)*n, :]
    I4 = LoffPoff[['ch0']].loc[(sampling_rate/2)*(n-1):(sampling_rate/2)*n, :]
    # 実験値αの配列
    alpha_exp = []
    for i in range(1000):
        alpha_i = 1-((I1.at[i, 'ch0']-I3.at[i, 'ch0'])/(I2.at[i, 'ch0']-I4.at[i, 'ch0']))
        alpha_exp.append(alpha_i)
    # αの理論値を求める、フィッティングを行って、温度と密度を出す
    lambda_theo, alpha_theo, temperature, gas_density = objective_function(alpha_exp)
    # plot
    plot_graph()
