# mcl
monte-calro-localizationのpython実装です.

## Description
実行するとロボットが横に動くシミュレータが起動します.
ロボットが移動するにつれてパーティクルが広がり, ランドマークを観測することで広がったパーティクルが改善されます.

![demo](https://github.com/kyo0221/mcl/blob/main/images/mcl_sample.gif)

## Parameters
|                      |                                    |     | 
| :------------------: | :--------------------------------: | --- | 
| time_interval        | 時刻の間隔[s]                      |     | 
| initial_pose         | ロボットの初期座標                 |     | 
| linear_vel           | 並進速度[m/s]                      |     | 
| linear_distance_std  | 直進1[m]で生じる道のりのばらつき   |     | 
| angular_distance_std | 回転1[rad]で生じる道のりのばらつき |     | 
| linear_rotation_std  | 直進1[m]で生じる向きのばらつき     |     | 
| angular_rotation_std | 回転1[rad]で生じる向きのばらつき   |     | 

## Reference
-   『詳解 確率ロボティクス ― Pythonによる基礎アルゴリズムの実装 ―』講談社〈KS理工学専門書〉、2019年、ISBN 978-406-51-7006-9