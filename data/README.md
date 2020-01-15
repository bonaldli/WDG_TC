# Data description
This is the implementation set I used.


## Prediction Result
<p align="center">
  <img src="https://github.com/bonaldli/Weakly_Corelated_LRTC/blob/master/figs/STN1_Prediction.png" /><img src="https://github.com/bonaldli/Weakly_Corelated_LRTC/blob/master/figs/STN2_Prediction.png" />
</p>

## Data Details
List of processed data:

----------------------------------------------
|<sub>Name </sub>|<sub>Description </sub>|<sub> Dimension </sub>|
|----------------|--------------------------------------------------------------|----------------------|
| A_m.npy | the full tensor of 15 mixed stations of 51 days | (15, 247, 51) |
| ohm_m.npy  | missing indicator, missing from t=74 at 51th day for ALL stations | (15, 247, 51) |
| poi_sim.npy | the POI similarity matrix of the 15 stations | (15, 15) |
| net.npy | the Neighbour Network* matrix of the 15 stations | (15, 15) |
| inflow_Tensor_87.npy | the full tensor of all 90 stations of 87 days(from 1st Jan to 28th Mar) | (87, 247, 90)|
| inflow_Tensor_4D.npy | the full tensor of all 90 stations of 12 weeks(from 2nd Jan to 26th Mar) | (12, 7, 247, 90) |

* comment: whether two stations are neighbour or not is defined by 'K-Hop Reachable' or not. I chose K=5, which means if 
travel from station A to station B takes no more that 5 stops, then they are neighbour.

## The selected 15 stations information
----------------------------------------------
|<sub>index in A_m </sub>|<sub>index in inflow_stn_name </sub>|<sub> real station code </sub>|<sub> station name </sub>|
|----------------|--------------------------------------------------------------|----------------------|----------------------|
|0	|51	|STN57	|LOHAS Park|
|1	|11	|STN12	|Choi Hung|
|2	|38 |STN39	|Hong Kong|
|3	|65	|STN81	|Sai Ying Pun|
|4	|84	|STN115	|Kam Sheung Road|
|5	|55	|STN68	|Sha Tin|
|6	|17	|STN18	|Cheung Sha Wan|
|7	|37	|STN38	|Lam Tin|
|8	|16	|STN17	|Sham Shui Po|
|9	|35	|STN36	|Heng Fa Chuen|
|10	|57	|STN71	|University|
|11	|56	|STN69	|Fo Tan|
|12	|54	|STN67	|Tai Wai|
|13	|64	|STN80	|East Tsim Sha Tsui|
|14	|87	|STN118	|Tin Shui Wai|

## The Training and The Testing 
----------------------------------------------
|<sub>Set</sub>|<sub>in Ohm</sub>|<sub>range</sub>|
|----------------|--------------------------------------------------------------|----------------------|
| Training | The **observed** entries | [:, :, 0:50] all the data of the first 50 days |
|  |  | [:, 0:75, 50] the data **before** timestamp = 75 in the last 51th day|
| Testing | The **unobserved** entries |  [:, 75:248, 50] the data **after** timestamp = 75 in the last 51th day|
