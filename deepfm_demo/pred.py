import requests
import json
import numpy as np

url = 'http://1899393106126557.cn-beijing.pai-eas.aliyuncs.com/api/predict/tf_server_simple/v1/models/model:predict'
headers = {'Content-Type': 'application/json','Authorization':'N2IxNWY2ZmFlMTliNDYxZGUwNzZiNDM4ZjY0OTVjODRkNjU2NjE3Zg=='}

# 准备输入数据
# 示例数据：一个形状为 [1, 28, 28, 1] 的张量
input_data =  [[0.5, 0.2], [0.1, 0.8],[0.0, 0.0],[1.0, 1.0],[2.0, 2.0]]

# 将输入数据转换为JSON格式
request_data = json.dumps({
    "inputs": input_data  # 将numpy数组转换为list
})

response = requests.post(url, data=request_data, headers=headers)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
    
    
import json
import base64

# 原始数据
data = {"inputs": [[0.5, 0.2], [0.1, 0.8]]}
data = {"inputs":
{
    "hour_num": [[
        5
    ]],
    "weekend": [[
        2
    ]],
    "slot_id": [[
        52
    ]],
    "native_slot_id": [[
        201
    ]],
    "ad_format": [[
        2
    ]],
    "campaign_id": [[
        33
    ]],
    "strategy_id": [[
        141
    ]],
    "creative_id": [[
        373
    ]],
    "os": [[
        1
    ]],
    "os_version": [[
        4
    ]],
    "country_id": [[
        0
    ]],
    "state_id": [[
        6
    ]],
    "city_id": [[
        62
    ]],
    "app_package": [[
        99
    ]],
    "model": [[
        115
    ]],
    "brand": [[
        2
    ]],
    "carrier": [[
        5
    ]],
    "net_kind": [[
        3
    ]],
    "slot_width": [[
        4
    ]],
    "slot_height": [[
        8
    ]],
    "screen_width": [[
        9
    ]],
    "screen_height": [[
        4
    ]],
    "brand_id": [[
        2
    ]],
    "model_id": [[
        101
    ]],
    "app_category_id": [[
        2
    ]],
    "f_code_size": [[
        2
    ]],
    "f_x": [[
        7
    ]],
    "f_y": [[
        6
    ]],
    "f_sys_platform": [[
        1
    ]],
    "f_industrys": [[
        0
    ]],
    "f_media_type": [[
        2
    ]],
    "bid_floor_price": [[
        0.00005383022774327122
    ]],
    "lat": [[
        0.026143790849673203
    ]],
    "lon": [[
        0.49575070821529743
    ]],
    "media_currency": [[
        0
    ]],
    "model_price": [[
        0
    ]],
    "sale_time": [[
        0.000017560667227307227
    ]],
    "creative_width": [[
        0
    ]],
    "creative_height": [[
        0
    ]],
    "creative_30d_ctr": [[
        0.4054
    ]],
    "creative_30d_cvr": [[
        0
    ]],
    "creative_7d_ctr": [[
        0.4558
    ]],
    "creative_7d_cvr": [[
        0
    ]],
    "creative_1d_ctr": [[
        0.4039
    ]],
    "creative_1d_cvr": [[
        0
    ]],
    "pkg_type_reqs_avg_15d": [[
        0.12492536834374565
    ]],
    "pkg_type_fills_avg_15d": [[
        0.051916204
    ]],
    "pkg_type_imps_avg_15d": [[
        0.018352380752258875
    ]],
    "pkg_type_clks_avg_15d": [[
        0.07432352538084445
    ]],
    "pkg_type_process_time_avg_15d": [[
        0.12456805970149254
    ]],
    "pkg_type_filter_media_income_avg_15d": [[
        0.015163456090651556
    ]],
    "pkg_type_filter_media_income_bid_amount_avg_15d": [[
        0.01462361111111111
    ]],
    "pkg_type_media_floor_avg_15d": [[
        0.013391675794085432
    ]],
    "pkg_type_media_bid_avg_15d": [[
        0.03207077922077922
    ]],
    "pkg_type_fill_ratio_15d": [[
        0.0492
    ]],
    "pkg_type_imp_ratio_15d": [[
        0.0112
    ]],
    "pkg_type_click_ratio_15d": [[
        0.10763333333333334
    ]],
    "pkg_type_media_cpm_15d": [[
        0.0024638190954773868
    ]],
    "pkg_type_media_cpc_15d": [[
        0.0010857142857142858
    ]],
    "pkg_type_media_second_price_ratio_15d": [[
        0.00897378640776699
    ]],
    "pkg_type_reqs_avg_7d": [[
        0.10131933333333333
    ]],
    "pkg_type_fills_avg_7d": [[
        0.027378917525773195
    ]],
    "pkg_type_imps_avg_7d": [[
        0.010675300108848772
    ]],
    "pkg_type_clks_avg_7d": [[
        0.06837797204339006
    ]],
    "pkg_type_process_time_avg_7d": [[
        0.1756567415730337
    ]],
    "pkg_type_filter_media_income_avg_7d": [[
        0.00898277894736842
    ]],
    "pkg_type_filter_media_income_bid_amount_avg_7d": [[
        0.009595507487520798
    ]],
    "pkg_type_media_floor_avg_7d": [[
        0.016187714285714288
    ]],
    "pkg_type_media_bid_avg_7d": [[
        0.025539444444444444
    ]],
    "pkg_type_fill_ratio_7d": [[
        0.0286
    ]],
    "pkg_type_imp_ratio_7d": [[
        0.0156
    ]],
    "pkg_type_click_ratio_7d": [[
        0.12503333333333333
    ]],
    "pkg_type_media_cpm_7d": [[
        0.0013524109014675053
    ]],
    "pkg_type_media_cpc_7d": [[
        0.0008625
    ]],
    "pkg_type_media_second_price_ratio_7d": [[
        0.008042608695652174
    ]],
    "pkg_type_reqs_avg_2d": [[
        0.09334576
    ]],
    "pkg_type_fills_avg_2d": [[
        0.049865650000000004
    ]],
    "pkg_type_imps_avg_2d": [[
        0.031049387504026388
    ]],
    "pkg_type_clks_avg_2d": [[
        0.13518149932643178
    ]],
    "pkg_type_process_time_avg_2d": [[
        0.0947869101978691
    ]],
    "pkg_type_filter_media_income_avg_2d": [[
        0.027298001480384904
    ]],
    "pkg_type_filter_media_income_bid_amount_avg_2d": [[
        0.02924192603441963
    ]],
    "pkg_type_media_floor_avg_2d": [[
        0.01230635103926097
    ]],
    "pkg_type_media_bid_avg_2d": [[
        0.02227336956521739
    ]],
    "pkg_type_fill_ratio_2d": [[
        0.0908
    ]],
    "pkg_type_imp_ratio_2d": [[
        0.017
    ]],
    "pkg_type_click_ratio_2d": [[
        0.12883333333333333
    ]],
    "pkg_type_media_cpm_2d": [[
        0.003708985507246377
    ]],
    "pkg_type_media_cpc_2d": [[
        0.0006
    ]],
    "pkg_type_media_second_price_ratio_2d": [[
        0.0046883248730964465
    ]],
    "pkg_type_wmdsp_fill_avg_15d": [[
        0.03986624968933808
    ]],
    "pkg_type_wmdsp_imp_avg_15d": [[
        0.1377216310556378
    ]],
    "pkg_type_wmdsp_clk_avg_15d": [[
        0.12095719578227415
    ]],
    "pkg_type_wmdsp_media_income_avg_15d": [[
        0.05008253968253968
    ]],
    "pkg_type_wmdsp_conv_avg_15d": [[
        0.00046725663716814164
    ]],
    "pkg_type_wmdsp_fill_avg_7d": [[
        0.04257410336255595
    ]],
    "pkg_type_wmdsp_imp_avg_7d": [[
        0.07677906619512022
    ]],
    "pkg_type_wmdsp_clk_avg_7d": [[
        0.2011615232930737
    ]],
    "pkg_type_wmdsp_media_income_avg_7d": [[
        0.07303333333333334
    ]],
    "pkg_type_wmdsp_conv_avg_7d": [[
        0.00036821705426356586
    ]],
    "pkg_type_wmdsp_fill_avg_2d": [[
        0.031084595422406155
    ]],
    "pkg_type_wmdsp_imp_avg_2d": [[
        0.08722404357241288
    ]],
    "pkg_type_wmdsp_clk_avg_2d": [[
        0.33381543942992875
    ]],
    "pkg_type_wmdsp_media_income_avg_2d": [[
        0.1239390243902439
    ]],
    "pkg_type_wmdsp_conv_avg_2d": [[
        0.0007123045744064852
    ]],
    "pkg_type_wmdsp_imp_ratio_15d": [[
        0.0113
    ]],
    "pkg_type_wmdsp_clk_ratio_15d": [[
        0.0004
    ]],
    "pkg_type_wmdsp_conv_ratio_15d": [[
        2.7353021626534194e-8
    ]],
    "pkg_type_wmdsp_cpm_15d": [[
        0.020027272727272727
    ]],
    "pkg_type_wmdsp_cpc_15d": [[
        0.00185
    ]],
    "pkg_type_wmdsp_cpa_15d": [[
        0.7967666666666666
    ]],
    "pkg_type_wmdsp_imp_ratio_7d": [[
        0.0159
    ]],
    "pkg_type_wmdsp_clk_ratio_7d": [[
        0.0018797029702970296
    ]],
    "pkg_type_wmdsp_conv_ratio_7d": [[
        7.741935483870967e-9
    ]],
    "pkg_type_wmdsp_cpm_7d": [[
        0.002863483146067416
    ]],
    "pkg_type_wmdsp_cpc_7d": [[
        0.00134
    ]],
    "pkg_type_wmdsp_cpa_7d": [[
        0.24276842105263155
    ]],
    "pkg_type_wmdsp_imp_ratio_2d": [[
        0.0169
    ]],
    "pkg_type_wmdsp_clk_ratio_2d": [[
        0.0034309734513274337
    ]],
    "pkg_type_wmdsp_conv_ratio_2d": [[
        1.3099507131044196e-8
    ]],
    "pkg_type_wmdsp_cpm_2d": [[
        0.0028635955056179776
    ]],
    "pkg_type_wmdsp_cpc_2d": [[
        0.00132
    ]],
    "pkg_type_wmdsp_cpa_2d": [[
        0.165232
    ]],
    "f_insert_way": [[
        1
    ]],
    "f_low_price": [[
        0.019230769230769232
    ]],
    "slot_reqs_avg_15d": [[
        0.028765567094547913
    ]],
    "slot_fills_avg_15d": [[
        0.06734975283513045
    ]],
    "slot_imps_avg_15d": [[
        0.05593982315218758
    ]],
    "slot_clks_avg_15d": [[
        0.10979647429157459
    ]],
    "slot_process_time_avg_15d": [[
        0.02786264855687606
    ]],
    "slot_filter_media_income_avg_15d": [[
        0.008376000658327847
    ]],
    "slot_filter_media_income_bid_amount_avg_15d": [[
        0.0075330499876607535
    ]],
    "slot_media_floor_avg_15d": [[
        0.06429216757741348
    ]],
    "slot_media_bid_avg_15d": [[
        0.0029504098360655737
    ]],
    "slot_fill_ratio_15d": [[
        0.1512
    ]],
    "slot_imp_ratio_15d": [[
        0.0039
    ]],
    "slot_click_ratio_15d": [[
        0.3232
    ]],
    "slot_media_cpm_15d": [[
        0.0061202020202020205
    ]],
    "slot_media_cpc_15d": [[
        0.0005
    ]],
    "slot_media_second_price_ratio_15d": [[
        0.00009840645915223627
    ]],
    "slot_reqs_avg_7d": [[
        0.02827974006309148
    ]],
    "slot_fills_avg_7d": [[
        0.037639470443349754
    ]],
    "slot_imps_avg_7d": [[
        0.04121561120661618
    ]],
    "slot_clks_avg_7d": [[
        0.09638004555066568
    ]],
    "slot_process_time_avg_7d": [[
        0.030415798414496036
    ]],
    "slot_filter_media_income_avg_7d": [[
        0.00753252391464312
    ]],
    "slot_filter_media_income_bid_amount_avg_7d": [[
        0.006785768940769081
    ]],
    "slot_media_floor_avg_7d": [[
        0.0633967332123412
    ]],
    "slot_media_bid_avg_7d": [[
        0.0023551601423487546
    ]],
    "slot_fill_ratio_7d": [[
        0.08525
    ]],
    "slot_imp_ratio_7d": [[
        0.0058
    ]],
    "slot_click_ratio_7d": [[
        0.3735
    ]],
    "slot_media_cpm_7d": [[
        0.006222303921568627
    ]],
    "slot_media_cpc_7d": [[
        0.0004533333333333333
    ]],
    "slot_media_second_price_ratio_7d": [[
        0.00005867308179285895
    ]],
    "slot_reqs_avg_2d": [[
        0.023449902114803627
    ]],
    "slot_fills_avg_2d": [[
        0.053261375796178345
    ]],
    "slot_imps_avg_2d": [[
        0.09279424189913842
    ]],
    "slot_clks_avg_2d": [[
        0.26422708923172195
    ]],
    "slot_process_time_avg_2d": [[
        0.01880141010575793
    ]],
    "slot_filter_media_income_avg_2d": [[
        0.021898024713099117
    ]],
    "slot_filter_media_income_bid_amount_avg_2d": [[
        0.019653860177937674
    ]],
    "slot_media_floor_avg_2d": [[
        0.05688646003262643
    ]],
    "slot_media_bid_avg_2d": [[
        0.004367320261437909
    ]],
    "slot_fill_ratio_2d": [[
        0.26935
    ]],
    "slot_imp_ratio_2d": [[
        0.0071
    ]],
    "slot_click_ratio_2d": [[
        0.3811
    ]],
    "slot_media_cpm_2d": [[
        0.005215909090909091
    ]],
    "slot_media_cpc_2d": [[
        0.00022
    ]],
    "slot_media_second_price_ratio_2d": [[
        0.00002723409278715618
    ]],
    "slot_wmdsp_fill_avg_15d": [[
        0.07072617067545714
    ]],
    "slot_wmdsp_imp_avg_15d": [[
        0.05833912890231621
    ]],
    "slot_wmdsp_clk_avg_15d": [[
        0.10341183521631303
    ]],
    "slot_wmdsp_media_income_avg_15d": [[
        0.0002834782608695652
    ]],
    "slot_wmdsp_conv_avg_15d": [[
        0.00045801671732522797
    ]],
    "slot_wmdsp_fill_avg_7d": [[
        0.04206005045664658
    ]],
    "slot_wmdsp_imp_avg_7d": [[
        0.0482636943257387
    ]],
    "slot_wmdsp_clk_avg_7d": [[
        0.091302541296061
    ]],
    "slot_wmdsp_media_income_avg_7d": [[
        0.00006597222222222222
    ]],
    "slot_wmdsp_conv_avg_7d": [[
        0.0003301777059773829
    ]],
    "slot_wmdsp_fill_avg_2d": [[
        0.06335746486486488
    ]],
    "slot_wmdsp_imp_avg_2d": [[
        0.1263647400003085
    ]],
    "slot_wmdsp_clk_avg_2d": [[
        0.32395394655065257
    ]],
    "slot_wmdsp_media_income_avg_2d": [[
        0.0002086092715231788
    ]],
    "slot_wmdsp_conv_avg_2d": [[
        0.0005763877606282155
    ]],
    "slot_wmdsp_imp_ratio_15d": [[
        0.0039
    ]],
    "slot_wmdsp_clk_ratio_15d": [[
        0.3239
    ]],
    "slot_wmdsp_conv_ratio_15d": [[
        6.776789495976282e-7
    ]],
    "slot_wmdsp_cpm_15d": [[
        0.0001165289256198347
    ]],
    "slot_wmdsp_cpc_15d": [[
        0
    ]],
    "slot_wmdsp_cpa_15d": [[
        0.0005
    ]],
    "slot_wmdsp_imp_ratio_7d": [[
        0.0059
    ]],
    "slot_wmdsp_clk_ratio_7d": [[
        0.3743
    ]],
    "slot_wmdsp_conv_ratio_7d": [[
        4.744525547445255e-7
    ]],
    "slot_wmdsp_cpm_7d": [[
        0.000005056179775280898
    ]],
    "slot_wmdsp_cpc_7d": [[
        0
    ]],
    "slot_wmdsp_cpa_7d": [[
        0.0001722222222222222
    ]],
    "slot_wmdsp_imp_ratio_2d": [[
        0.007
    ]],
    "slot_wmdsp_clk_ratio_2d": [[
        0.3818
    ]],
    "slot_wmdsp_conv_ratio_2d": [[
        7.556131260794473e-8
    ]],
    "slot_wmdsp_cpm_2d": [[
        0.000004269662921348315
    ]],
    "slot_wmdsp_cpc_2d": [[
        0
    ]],
    "slot_wmdsp_cpa_2d": [[
        0.0004228571428571429
    ]],
    "gender": [[
        1
    ]],
    "age": [[
        0
    ]],
    "gender_tag": [[
        1
    ]],
    "age_tag": [[
        0
    ]],
    "education_level": [[
        0
    ]],
    "marital_status": [[
        0
    ]],
    "education": [[
        0
    ]],
    "consumptionlevel": [[
        0
    ]],
    "life_stage": [[
        0
    ]],
    "adjustment_rate_signal_1": [[
        0
    ]],
    "adjustment_rate_signal_2": [[
        0
    ]],
    "adjustment_rate_signal_3": [[
        0
    ]],
    "adjustment_rate_signal_4": [[
        0
    ]],
    "order_rate_signal_1": [[
        0
    ]],
    "order_rate_signal_2": [[
        0
    ]],
    "order_rate_signal_3": [[
        0
    ]],
    "order_rate_signal_4": [[
        0
    ]],
    "first_wake_rate_signal_1": [[
        0
    ]],
    "first_wake_rate_signal_2": [[
        0
    ]],
    "app_1": [[
        1
    ]],
    "app_2": [[
        0
    ]],
    "app_3": [[
        0
    ]],
    "app_4": [[
        0
    ]],
    "app_5": [[
        0
    ]],
    "app_6": [[
        0
    ]],
    "app_7": [[
        0
    ]],
    "app_8": [[
        0
    ]],
    "app_9": [[
        0
    ]],
    "app_10": [[
        0
    ]],
    "app_11": [[
        0
    ]],
    "app_12": [[
        0
    ]],
    "app_13": [[
        0
    ]],
    "app_14": [[
        0
    ]],
    "app_15": [[
        0
    ]],
    "app_16": [[
        0
    ]],
    "app_17": [[
        0
    ]],
    "app_18": [[
        0
    ]],
    "app_19": [[
        0
    ]],
    "app_20": [[
        0
    ]],
    "app_21": [[
        0
    ]],
    "app_22": [[
        0
    ]],
    "app_23": [[
        0
    ]],
    "app_24": [[
        0
    ]],
    "app_25": [[
        0
    ]],
    "app_26": [[
        0
    ]],
    "app_27": [[
        0
    ]],
    "app_28": [[
        0
    ]],
    "app_29": [[
        0
    ]],
    "app_30": [[
        0
    ]],
    "app_31": [[
        0
    ]],
    "app_32": [[
        0
    ]],
    "app_33": [[
        0
    ]],
    "app_34": [[
        1
    ]],
    "app_35": [[
        1
    ]],
    "app_36": [[
        0
    ]],
    "app_37": [[
        0
    ]],
    "app_38": [[
        0
    ]],
    "app_39": [[
        0
    ]],
    "app_40": [[
        0
    ]],
    "app_41": [[
        0
    ]],
    "app_42": [[
        0
    ]],
    "app_43": [[
        0
    ]],
    "app_44": [[
        0
    ]],
    "app_45": [[
        0
    ]],
    "app_46": [[
        0
    ]],
    "app_47": [[
        0
    ]],
    "app_48": [[
        0
    ]],
    "app_49": [[
        0
    ]],
    "app_50": [[
        0
    ]],
    "app_51": [[
        0
    ]],
    "app_52": [[
        0
    ]],
    "app_53": [[
        0
    ]],
    "app_54": [[
        0
    ]],
    "app_55": [[
        0
    ]],
    "app_56": [[
        0
    ]],
    "app_57": [[
        0
    ]],
    "app_58": [[
        0
    ]],
    "app_59": [[
        0
    ]],
    "app_60": [[
        0
    ]],
    "app_61": [[
        0
    ]],
    "app_62": [[
        0
    ]],
    "app_63": [[
        0
    ]],
    "app_64": [[
        1
    ]],
    "app_65": [[
        0
    ]],
    "app_66": [[
        0
    ]],
    "app_67": [[
        0
    ]],
    "app_68": [[
        0
    ]],
    "app_69": [[
        0
    ]],
    "app_70": [[
        0
    ]],
    "app_71": [[
        0
    ]],
    "app_72": [[
        0
    ]],
    "app_73": [[
        0
    ]],
    "app_74": [[
        0
    ]],
    "app_75": [[
        0
    ]],
    "app_76": [[
        0
    ]],
    "app_77": [[
        0
    ]],
    "app_78": [[
        0
    ]],
    "app_79": [[
        0
    ]],
    "app_80": [[
        0
    ]],
    "app_81": [[
        0
    ]],
    "app_82": [[
        0
    ]],
    "app_83": [[
        0
    ]],
    "app_84": [[
        0
    ]],
    "app_85": [[
        0
    ]],
    "app_86": [[
        0
    ]],
    "app_87": [[
        0
    ]],
    "app_88": [[
        0
    ]],
    "app_89": [[
        0
    ]],
    "app_90": [[
        0
    ]],
    "app_91": [[
        0
    ]],
    "app_92": [[
        0
    ]],
    "app_93": [[
        0
    ]],
    "app_94": [[
        0
    ]],
    "app_95": [[
        0
    ]],
    "app_96": [[
        0
    ]],
    "app_97": [[
        0
    ]],
    "app_98": [[
        0
    ]],
    "app_99": [[
        0
    ]],
    "app_100": [[
        0
    ]],
    "app_101": [[
        0
    ]],
    "app_102": [[
        0
    ]],
    "app_103": [[
        0
    ]],
    "app_104": [[
        0
    ]],
    "app_105": [[
        0
    ]],
    "app_106": [[
        0
    ]],
    "app_107": [[
        0
    ]],
    "app_108": [[
        0
    ]],
    "app_109": [[
        0
    ]],
    "app_110": [[
        1
    ]],
    "app_111": [[
        0
    ]],
    "app_112": [[
        0
    ]],
    "app_113": [[
        0
    ]],
    "app_114": [[
        0
    ]],
    "app_115": [[
        0
    ]],
    "app_116": [[
        0
    ]],
    "app_117": [[
        0
    ]],
    "app_118": [[
        0
    ]],
    "app_119": [[
        0
    ]],
    "app_120": [[
        0
    ]],
    "app_121": [[
        0
    ]],
    "app_122": [[
        0
    ]],
    "app_123": [[
        0
    ]],
    "app_124": [[
        0
    ]],
    "app_125": [[
        0
    ]],
    "app_126": [[
        0
    ]],
    "app_127": [[
        0
    ]],
    "app_128": [[
        0
    ]],
    "app_129": [[
        0
    ]],
    "app_130": [[
        0
    ]],
    "app_131": [[
        0
    ]],
    "app_132": [[
        0
    ]],
    "app_133": [[
        0
    ]],
    "app_134": [[
        0
    ]],
    "app_135": [[
        0
    ]],
    "app_136": [[
        0
    ]],
    "app_137": [[
        0
    ]],
    "app_138": [[
        0
    ]],
    "app_139": [[
        0
    ]],
    "app_140": [[
        0
    ]],
    "app_141": [[
        0
    ]],
    "app_142": [[
        0
    ]],
    "app_143": [[
        0
    ]],
    "app_144": [[
        0
    ]],
    "app_145": [[
        0
    ]],
    "app_146": [[
        0
    ]],
    "app_147": [[
        0
    ]],
    "app_148": [[
        0
    ]],
    "app_149": [[
        0
    ]],
    "app_150": [[
        0
    ]],
    "app_151": [[
        0
    ]],
    "app_152": [[
        0
    ]],
    "app_153": [[
        0
    ]],
    "app_154": [[
        0
    ]],
    "app_155": [[
        0
    ]],
    "app_156": [[
        0
    ]],
    "app_157": [[
        0
    ]],
    "app_158": [[
        0
    ]],
    "app_159": [[
        0
    ]],
    "app_160": [[
        0
    ]],
    "app_161": [[
        0
    ]],
    "app_162": [[
        0
    ]],
    "app_163": [[
        0
    ]],
    "app_164": [[
        0
    ]],
    "app_165": [[
        0
    ]],
    "app_166": [[
        0
    ]],
    "app_167": [[
        0
    ]],
    "app_168": [[
        0
    ]],
    "app_169": [[
        0
    ]],
    "app_170": [[
        0
    ]],
    "app_171": [[
        0
    ]],
    "app_172": [[
        0
    ]],
    "app_173": [[
        0
    ]],
    "app_174": [[
        0
    ]],
    "app_175": [[
        0
    ]],
    "app_176": [[
        0
    ]],
    "app_177": [[
        0
    ]],
    "app_178": [[
        0
    ]],
    "app_179": [[
        0
    ]],
    "app_180": [[
        0
    ]],
    "app_181": [[
        0
    ]],
    "app_182": [[
        0
    ]],
    "app_183": [[
        0
    ]],
    "app_184": [[
        0
    ]],
    "app_185": [[
        0
    ]],
    "app_186": [[
        0
    ]],
    "app_187": [[
        0
    ]],
    "app_188": [[
        0
    ]],
    "app_189": [[
        0
    ]],
    "app_190": [[
        0
    ]],
    "app_191": [[
        0
    ]],
    "app_192": [[
        0
    ]],
    "app_193": [[
        0
    ]],
    "app_194": [[
        0
    ]],
    "app_195": [[
        0
    ]],
    "app_196": [[
        0
    ]],
    "app_197": [[
        0
    ]],
    "app_198": [[
        0
    ]],
    "app_199": [[
        0
    ]],
    "app_200": [[
        0
    ]],
    "pkg_1": [[
        0
    ]],
    "pkg_2": [[
        0
    ]],
    "pkg_3": [[
        0
    ]],
    "pkg_4": [[
        0
    ]],
    "pkg_5": [[
        0
    ]],
    "pkg_6": [[
        0
    ]],
    "pkg_7": [[
        0
    ]],
    "pkg_8": [[
        0
    ]],
    "pkg_9": [[
        1
    ]],
    "pkg_10": [[
        0
    ]],
    "pkg_11": [[
        0
    ]],
    "pkg_12": [[
        0
    ]],
    "pkg_13": [[
        0
    ]],
    "pkg_14": [[
        0
    ]],
    "pkg_15": [[
        1
    ]],
    "pkg_16": [[
        0
    ]],
    "pkg_17": [[
        0
    ]],
    "pkg_18": [[
        0
    ]],
    "pkg_19": [[
        1
    ]],
    "pkg_20": [[
        0
    ]],
    "pkg_21": [[
        0
    ]],
    "pkg_22": [[
        0
    ]],
    "pkg_23": [[
        0
    ]],
    "pkg_24": [[
        0
    ]],
    "pkg_25": [[
        0
    ]],
    "pkg_26": [[
        0
    ]],
    "pkg_27": [[
        0
    ]],
    "pkg_28": [[
        0
    ]],
    "pkg_29": [[
        0
    ]],
    "pkg_30": [[
        0
    ]],
    "pkg_31": [[
        0
    ]],
    "pkg_32": [[
        0
    ]],
    "pkg_33": [[
        0
    ]],
    "pkg_34": [[
        0
    ]],
    "pkg_35": [[
        0
    ]],
    "pkg_36": [[
        0
    ]],
    "pkg_37": [[
        0
    ]],
    "pkg_38": [[
        0
    ]],
    "pkg_39": [[
        0
    ]],
    "pkg_40": [[
        0
    ]],
    "pkg_41": [[
        0
    ]],
    "pkg_42": [[
        0
    ]],
    "pkg_43": [[
        0
    ]],
    "pkg_44": [[
        0
    ]],
    "pkg_45": [[
        0
    ]],
    "pkg_46": [[
        0
    ]],
    "pkg_47": [[
        0
    ]],
    "pkg_48": [[
        0
    ]],
    "pkg_49": [[
        0
    ]],
    "pkg_50": [[
        0
    ]],
    "pkg_51": [[
        0
    ]],
    "pkg_52": [[
        0
    ]],
    "pkg_53": [[
        0
    ]],
    "pkg_54": [[
        0
    ]],
    "pkg_55": [[
        0
    ]],
    "pkg_56": [[
        0
    ]],
    "pkg_57": [[
        0
    ]],
    "pkg_58": [[
        0
    ]],
    "pkg_59": [[
        0
    ]],
    "pkg_60": [[
        0
    ]],
    "pkg_61": [[
        0
    ]],
    "pkg_62": [[
        0
    ]],
    "pkg_63": [[
        0
    ]],
    "pkg_64": [[
        0
    ]],
    "pkg_65": [[
        0
    ]],
    "pkg_66": [[
        0
    ]],
    "pkg_67": [[
        0
    ]],
    "pkg_68": [[
        0
    ]],
    "pkg_69": [[
        0
    ]],
    "pkg_70": [[
        0
    ]],
    "pkg_71": [[
        0
    ]],
    "pkg_72": [[
        0
    ]],
    "pkg_73": [[
        0
    ]],
    "pkg_74": [[
        0
    ]],
    "pkg_75": [[
        0
    ]],
    "pkg_76": [[
        0
    ]],
    "pkg_77": [[
        0
    ]],
    "pkg_78": [[
        0
    ]],
    "pkg_79": [[
        0
    ]],
    "pkg_80": [[
        0
    ]],
    "pkg_81": [[
        0
    ]],
    "pkg_82": [[
        0
    ]],
    "pkg_83": [[
        0
    ]],
    "pkg_84": [[
        0
    ]],
    "pkg_85": [[
        0
    ]],
    "pkg_86": [[
        0
    ]],
    "pkg_87": [[
        0
    ]],
    "pkg_88": [[
        0
    ]],
    "pkg_89": [[
        1
    ]],
    "pkg_90": [[
        0
    ]],
    "pkg_91": [[
        0
    ]],
    "pkg_92": [[
        0
    ]],
    "pkg_93": [[
        0
    ]],
    "pkg_94": [[
        0
    ]],
    "pkg_95": [[
        0
    ]],
    "pkg_96": [[
        0
    ]],
    "pkg_97": [[
        0
    ]],
    "pkg_98": [[
        0
    ]],
    "pkg_99": [[
        0
    ]],
    "pkg_100": [[
        0
    ]],
    "three_days_reqs": [[
        0.013246572158958867
    ]],
    "three_days_imps": [[
        0.013246572158958867
    ]],
    "three_days_clks": [[
        0
    ]],
    "three_days_cuhuos": [[
        0
    ]],
    "three_days_cilius": [[
        0
    ]],
    "three_days_actives": [[
        0
    ]],
    "three_days_event_lens": [[
        0.4
    ]],
    "three_days_hours": [[
        0.3333333333333333
    ]],
    "three_days_slots": [[
        0.18181818181818182
    ]],
    "three_days_ad_formats": [[
        0.5
    ]],
    "three_days_campaign_ids": [[
        0.15384615384615385
    ]],
    "three_days_creatives": [[
        0.008563273073263558
    ]],
    "three_days_apps": [[
        0.06944444444444445
    ]],
    "three_days_citys": [[
        0.01282051282051282
    ]],
    "three_days_avg_bid_price": [[
        0.00006971563981042654
    ]],
    "three_days_nets": [[
        0.14285714285714285
    ]],
    "medsuc_reqs_3d": [[
        0.00018638992772880867
    ]],
    "hour00_reqs_3d": [[
        0
    ]],
    "hour01_reqs_3d": [[
        0
    ]],
    "hour02_reqs_3d": [[
        0
    ]],
    "hour03_reqs_3d": [[
        0
    ]],
    "hour04_reqs_3d": [[
        0
    ]],
    "hour05_reqs_3d": [[
        0
    ]],
    "hour06_reqs_3d": [[
        0.00028814291888776836
    ]],
    "hour07_reqs_3d": [[
        0
    ]],
    "hour08_reqs_3d": [[
        0
    ]],
    "hour09_reqs_3d": [[
        0.002333450005833625
    ]],
    "hour10_reqs_3d": [[
        0.0007254261878853826
    ]],
    "hour11_reqs_3d": [[
        0.00004383273428596476
    ]],
    "hour12_reqs_3d": [[
        0
    ]],
    "hour13_reqs_3d": [[
        0
    ]],
    "hour14_reqs_3d": [[
        0
    ]],
    "hour15_reqs_3d": [[
        0
    ]],
    "hour16_reqs_3d": [[
        0
    ]],
    "hour17_reqs_3d": [[
        0
    ]],
    "hour18_reqs_3d": [[
        0
    ]],
    "hour19_reqs_3d": [[
        0
    ]],
    "hour20_reqs_3d": [[
        0
    ]],
    "hour21_reqs_3d": [[
        0
    ]],
    "hour22_reqs_3d": [[
        0
    ]],
    "hour23_reqs_3d": [[
        0
    ]],
    "hours_3d": [[
        0.125
    ]],
    "medias_3d": [[
        0.07142857142857142
    ]],
    "slots_3d": [[
        0.03896103896103896
    ]],
    "ad_formats_3d": [[
        0.5
    ]],
    "pkgs_3d": [[
        0.0196078431372549
    ]],
    "thirdplatformkeys_3d": [[
        0.13333333333333333
    ]],
    "mediafloor_sum_3d": [[
        1.852941176470588e-7
    ]],
    "mediafloor_avg_3d": [[
        0.0000039347315331623675
    ]],
    "mediafloor_max_3d": [[
        0.000032734295721627547
    ]],
    "mediafloor_min_3d": [[
        0.0000010666666666666667
    ]],
    "mediafloorprice_adtype2_sum_3d": [[
        4.7058825490196084e-8
    ]],
    "mediafloorprice_adtype2_avg_3d": [[
        9.834130990624794e-7
    ]],
    "mediafloorprice_adtype2_max_3d": [[
        9.820288716488265e-7
    ]],
    "mediafloorprice_adtype2_min_3d": [[
        9.834130990624794e-7
    ]],
    "mediafloorprice_adtype3_sum_3d": [[
        4.0006435077642275e-7
    ]],
    "mediafloorprice_adtype3_avg_3d": [[
        0.00001275
    ]],
    "mediafloorprice_adtype3_max_3d": [[
        0.00006666666666666667
    ]],
    "mediafloorprice_adtype3_min_3d": [[
        0.0000013333333333333334
    ]],
    "mediafloorprice_adtype4_sum_3d": [[
        0
    ]],
    "mediafloorprice_adtype4_avg_3d": [[
        0
    ]],
    "mediafloorprice_adtype4_max_3d": [[
        0
    ]],
    "mediafloorprice_adtype4_min_3d": [[
        0
    ]],
    "mediafloorprice_adtype56_sum_3d": [[
        0
    ]],
    "mediafloorprice_adtype56_avg_3d": [[
        0
    ]],
    "mediafloorprice_adtype56_max_3d": [[
        0
    ]],
    "mediafloorprice_adtype56_min_3d": [[
        0
    ]],
    "dsporiginalbidprice_sum_3d": [[
        0.000004025523304752826
    ]],
    "dsporiginalbidprice_avg_3d": [[
        0.00006221153846153846
    ]],
    "dsporiginalbidprice_max_3d": [[
        0.00025
    ]],
    "dsporiginalbidprice_min_3d": [[
        0.00002012072434607646
    ]],
    "dsporiginalbidprice_adtype2_sum_3d": [[
        0.0000034365721773457843
    ]],
    "dsporiginalbidprice_adtype2_avg_3d": [[
        0.00012716666666666668
    ]],
    "dsporiginalbidprice_adtype2_max_3d": [[
        0.00037060908797937475
    ]],
    "dsporiginalbidprice_adtype2_min_3d": [[
        0.000026666666666666667
    ]],
    "dsporiginalbidprice_adtype3_sum_3d": [[
        0.0000046554782371401275
    ]],
    "dsporiginalbidprice_adtype3_avg_3d": [[
        0.00021786052382455034
    ]],
    "dsporiginalbidprice_adtype3_max_3d": [[
        0.000758495145631068
    ]],
    "dsporiginalbidprice_adtype3_min_3d": [[
        0.000029999999999999997
    ]],
    "dsporiginalbidprice_adtype4_sum_3d": [[
        0
    ]],
    "dsporiginalbidprice_adtype4_avg_3d": [[
        0
    ]],
    "dsporiginalbidprice_adtype4_max_3d": [[
        0
    ]],
    "dsporiginalbidprice_adtype4_min_3d": [[
        0
    ]],
    "dsporiginalbidprice_adtype56_sum_3d": [[
        0
    ]],
    "dsporiginalbidprice_adtype56_avg_3d": [[
        0
    ]],
    "dsporiginalbidprice_adtype56_max_3d": [[
        0
    ]],
    "dsporiginalbidprice_adtype56_min_3d": [[
        0
    ]],
    "dsporiginalbidprice_thirdplatform359_sum_3d": [[
        0.0000055703820722116035
    ]],
    "dsporiginalbidprice_thirdplatform359_avg_3d": [[
        0.00028706108706108704
    ]],
    "dsporiginalbidprice_thirdplatform359_max_3d": [[
        0.0005531505531505531
    ]],
    "dsporiginalbidprice_thirdplatform359_min_3d": [[
        0.00023088023088023088
    ]],
    "dsporiginalbidprice_thirdplatform398_sum_3d": [[
        8.475250340936233e-7
    ]],
    "dsporiginalbidprice_thirdplatform398_avg_3d": [[
        0.0002323478260869565
    ]],
    "dsporiginalbidprice_thirdplatform398_max_3d": [[
        0.00031304347826086954
    ]],
    "dsporiginalbidprice_thirdplatform398_min_3d": [[
        0.0001391304347826087
    ]],
    "dsporiginalbidprice_thirdplatform375_sum_3d": [[
        0
    ]],
    "dsporiginalbidprice_thirdplatform375_avg_3d": [[
        0
    ]],
    "dsporiginalbidprice_thirdplatform375_max_3d": [[
        0
    ]],
    "dsporiginalbidprice_thirdplatform375_min_3d": [[
        0
    ]],
    "dsporiginalbidprice_thirdplatform368_sum_3d": [[
        0
    ]],
    "dsporiginalbidprice_thirdplatform368_avg_3d": [[
        0
    ]],
    "dsporiginalbidprice_thirdplatform368_max_3d": [[
        0
    ]],
    "dsporiginalbidprice_thirdplatform368_min_3d": [[
        0
    ]],
    "dsporiginalbidprice_thirdplatform310_sum_3d": [[
        0
    ]],
    "dsporiginalbidprice_thirdplatform310_avg_3d": [[
        0
    ]],
    "dsporiginalbidprice_thirdplatform310_max_3d": [[
        0
    ]],
    "dsporiginalbidprice_thirdplatform310_min_3d": [[
        0
    ]],
    "imps_3d": [[
        0.0025493687277436065
    ]],
    "clks_3d": [[
        0
    ]],
    "winprice_sum_3d": [[
        0.000210504109186968
    ]],
    "winprice_avg_3d": [[
        0.00008643198174706648
    ]],
    "winprice_max_3d": [[
        0.0005362776025236593
    ]],
    "winprice_min_3d": [[
        0.000007822685788787483
    ]],
    "winprice_adtype2_sum_3d": [[
        0.002136195221668583
    ]],
    "winprice_adtype2_avg_3d": [[
        0.0003669036137440758
    ]],
    "winprice_adtype2_max_3d": [[
        0.00045105074320861096
    ]],
    "winprice_adtype2_min_3d": [[
        0.00016473459426479561
    ]],
    "winprice_adtype3_sum_3d": [[
        0.000041187159297395516
    ]],
    "winprice_adtype3_avg_3d": [[
        0.0009916666666666665
    ]],
    "winprice_adtype3_max_3d": [[
        0.0008585858585858586
    ]],
    "winprice_adtype3_min_3d": [[
        0.0009916666666666665
    ]],
    "winprice_adtype4_sum_3d": [[
        0
    ]],
    "winprice_adtype4_avg_3d": [[
        0
    ]],
    "winprice_adtype4_max_3d": [[
        0
    ]],
    "winprice_adtype4_min_3d": [[
        0
    ]],
    "winprice_adtype56_sum_3d": [[
        0.0000993112698944979
    ]],
    "winprice_adtype56_avg_3d": [[
        0.000025912572104551598
    ]],
    "winprice_adtype56_max_3d": [[
        0.0000405588102748986
    ]],
    "winprice_adtype56_min_3d": [[
        0.0000067598017124831005
    ]],
    "dspwinprice_sum_3d": [[
        0.00016755033264576858
    ]],
    "dspwinprice_avg_3d": [[
        0.00004667507920930885
    ]],
    "dspwinprice_max_3d": [[
        0.0003043848964677223
    ]],
    "dspwinprice_min_3d": [[
        0.000004205804009533156
    ]],
    "dspwinprice_adtype2_sum_3d": [[
        0.0015467160037002773
    ]],
    "dspwinprice_adtype2_avg_3d": [[
        0.00020651327109036345
    ]],
    "dspwinprice_adtype2_max_3d": [[
        0.0002934311437145715
    ]],
    "dspwinprice_adtype2_min_3d": [[
        0.00009003001000333445
    ]],
    "dspwinprice_adtype3_sum_3d": [[
        0.00003485452871767692
    ]],
    "dspwinprice_adtype3_avg_3d": [[
        0.0008332777592530844
    ]],
    "dspwinprice_adtype3_max_3d": [[
        0.000816933638443936
    ]],
    "dspwinprice_adtype3_min_3d": [[
        0.0008332777592530844
    ]],
    "dspwinprice_adtype4_sum_3d": [[
        0
    ]],
    "dspwinprice_adtype4_avg_3d": [[
        0
    ]],
    "dspwinprice_adtype4_max_3d": [[
        0
    ]],
    "dspwinprice_adtype4_min_3d": [[
        0
    ]],
    "dspwinprice_adtype56_sum_3d": [[
        0.00007727354836119864
    ]],
    "dspwinprice_adtype56_avg_3d": [[
        0.000014007307551766138
    ]],
    "dspwinprice_adtype56_max_3d": [[
        0.00002192448233861145
    ]],
    "dspwinprice_adtype56_min_3d": [[
        0.0000036540803897685746
    ]],
    "dspwinprice_thirdplatform359_sum_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform359_avg_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform359_max_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform359_min_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform398_sum_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform398_avg_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform398_max_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform398_min_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform375_sum_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform375_avg_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform375_max_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform375_min_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform368_sum_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform368_avg_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform368_max_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform368_min_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform310_sum_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform310_avg_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform310_max_3d": [[
        0
    ]],
    "dspwinprice_thirdplatform310_min_3d": [[
        0
    ]]
}}

# 将数据转换为JSON字符串
json_str = json.dumps(data)

# 将字符串转换为字节串
json_bytes = json_str.encode('utf-8')

# 对字节串进行Base64编码
encoded_bytes = base64.b64encode(json_bytes)

# 将编码结果转换回字符串
encoded_str = encoded_bytes.decode('utf-8')

print(encoded_str)