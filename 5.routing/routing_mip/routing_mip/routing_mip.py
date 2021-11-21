
from IPython.core.display import display
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from mip import * 
#import pulp
from itertools import product, combinations_with_replacement
from joblib import Parallel, delayed
from numpy.random import default_rng
from scipy.stats import norm, gamma
import logging

#np.random.seed(10)
seed = 10 #10であるに理由は特にないが、この時いい感じに配送先がばらけて見やすい

num_places = 10  # 地点の数
num_days = 30  # 計画の対象日数
num_requests = 120  # 荷物の数

mean_travel_time_to_destinations = 100  # 自社から平均的に100分程度距離に配達先候補があるとしてデータを作る．
H_regular = 8*60  # 8時間が定時労働
H_max_overtime = 3*60  # 残業3時間まで
c = 3000//60  # 残業による経費60分 3000円
W = 4000  # 4 トントラックを利用
delivery_outsourcing_unit_cost = 4600  # 100kg あたり4600円の配送費用
delivery_time_window = 3  # 連続する3日が配達可能な候補日となる．
avg_weight = 1000  # 荷物の平均的な重さを1000kgとする

K = range(num_places)  # 地点の集合
o = 0  # 自社拠点を表す地点
K_minus_o = K[1:]  # 配達先の集合
rng = default_rng(seed) #numpy.random.Generatorによる疑似乱数の生成


_K = norm.rvs(loc=0, scale=mean_travel_time_to_destinations, size=(len(K), 2), random_state=seed)
#_K = rng.normal(0, mean_travel_time_to_destinations, size=(len(K), 2))  # 各地点の座標を設定
_K[o,:] = 0  # 自社拠点は原点とする．
t = np.array([[np.floor(np.linalg.norm(_K[k] - _K[l])) for k in K] for l in K])  # 各地点間の移動時間行列(分)

D = range(num_days)  # 日付の集合
R = range(num_requests)  # 荷物の集合
k = rng.choice(K_minus_o, size=len(R))
#k = np.random.choice(K_minus_o, size=len(R))  # k[r] は 荷物 r の配送先を表す
d_0 = rng.choice(D, size=len(R))
#d_0 = np.random.choice(D, size=len(R))  # d_0[r] は荷物 r の配送可能日の初日を表す
d_1 = d_0 + delivery_time_window-1   # d_1[r] は荷物 r の配送可能日の最終日を表す
w = np.floor(gamma.rvs(10, scale=avg_weight/10, size=len(R), random_state=seed))
#w = np.floor(np.random.gamma(10, avg_weight/10, size=len(R)))  # w[r] が荷物 r の重さ(kg)を表す
f = np.ceil(w/100)*delivery_outsourcing_unit_cost  # f[r] が荷物 r の外部委託時の配送料を表す


def simulate_route(z):
    # enumerate_routes の中でのみ用いる関数
    # z は k_minus_o の部分集合を意味する長さnum_placesの0または1の値のリストで、
    # z[k] == 1 (k in K) が k への訪問があることを意味する．

    if z[0] == 0:  # 自社拠点を通らない移動経路は不適切なので None を返し，後段で除去する．
        return None

    # 巡回セールスマン問題を解く
    daily_route_prob = Model()#pulp.LpProblem(sense=pulp.LpMinimize)

    # k から l への移動の有無
    x = {
        (k, l): 
            daily_route_prob.add_var(name=f'x_{k}_{l}', var_type=BINARY) if k != l else LinExpr(0)
        for k, l in product(K, K)
    }
    
    #print(x)

    # MTZ 定式化のための補助変数
    u = {
        k: daily_route_prob.add_var(
            name=f'u_{k}', 
            lb=1, 
            ub=len(K) - 1,
        )
        for k in K_minus_o
    }
    # MTZ 定式化の補助変数の説明では訪問順序であることを意識して u[0] を変数かのように書いたが，
    # 実際には 0 に固定されている値であるので，ここでは変数としては u[0] は定義しない．    

    h = daily_route_prob.add_var(f'h', lb=0, var_type=CONTINUOUS)

    # 移動の構造
    for l in K:
        daily_route_prob += (
            xsum([x[k,l] for k in K]) <= 1
        )

    for l in K:
        if z[l] == 1:
            # z で l への訪問が指定されている場合，必ず訪問するようにする．
            daily_route_prob += (
                xsum([x[k,l] for k in K]) == 1
            )
            daily_route_prob += (
                xsum([x[l,k] for k in K]) == 1
            )

        else:
            # z で l への訪問が禁止されている場合，訪問ができないように x に制約を入れる
            daily_route_prob += (
                xsum([x[k,l] for k in K]) == 0
            )
            daily_route_prob += (
                xsum([x[l,k] for k in K]) == 0                
            )
            

    # サイクルの除去．
    for k, l in product(K_minus_o, K_minus_o):
        daily_route_prob += (
            u[k] + 1 <= u[l] + len(K_minus_o) * (1 - x[k, l])
        )    

    # 労務関係．(巡回セールスマン問題にはない制約だが，これが満たされない場合実行不可能としたいので追加)
    travel = xsum([t[k, l]*x[k, l] for k, l in product(K, K)]) # 移動時間
    daily_route_prob += (travel - H_regular <= h)
    daily_route_prob += (h <= H_max_overtime)

    # 目的関数
    daily_route_prob.objective = minimize(travel)
    status = daily_route_prob.optimize()

    return {
        'z': z,
        'route': { # k から l への移動の有無を辞書で保持
            (k, l): x[k, l].x
            for k, l in product(K, K)
        },
        'optimal': OptimizationStatus(status),
        '移動時間': travel.x,
        '残業時間': h.x,
    }


u = 0
def enumerate_routes():
    # 移動経路を列挙する    
    # joblib を用いて計算を並列化(16並列)して，K_minus_o のすべての部分集合に対する最短の移動経路を列挙
    # これは次のコードを並列化したもの．
    # routes = []
    # for z in product([0,1], repeat=len(K)):
    #     routes.append(simulate_route(z))
    routes = Parallel(n_jobs=16)(
        [delayed(simulate_route)(z) for z in product([0,1], repeat=len(K))]
    )
    
    # 結果が None のもの（自社拠点を通らないもの）を除去
    routes = pd.DataFrame(filter(lambda x: x is not None, routes))
    
    # 結果が Optimal でないもの（ここでは移動時間が長すぎて実行不能となるもの）を除去
    u = routes = routes[routes.optimal == OptimizationStatus.OPTIMAL].copy()
    return u

logging.basicConfig(filename='debug.log', level=logging.DEBUG)
logging.debug(f'shogommm: {enumerate_routes().optimal}')
routes_df = enumerate_routes()