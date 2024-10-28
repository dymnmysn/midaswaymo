waymo2kitti= {
    'TYPE_UNDEFINED': "unlabeled",
    'TYPE_CAR': "car",
    'TYPE_TRUCK': "truck",
    'TYPE_BUS': "bus",
    'TYPE_OTHER_VEHICLE': "other-vehicle",
    'TYPE_MOTORCYCLIST': "motorcyclist",
    'TYPE_BICYCLIST': "bicyclist",
    'TYPE_PEDESTRIAN': "person",
    'TYPE_SIGN': "traffic-sign",
    'TYPE_TRAFFIC_LIGHT': "traffic-sign",
    'TYPE_POLE': "pole",
    'TYPE_CONSTRUCTION_CONE': "pole",
    'TYPE_BICYCLE': "bicycle",
    'TYPE_MOTORCYCLE': "motorcycle",
    'TYPE_BUILDING': "building",
    'TYPE_VEGETATION': "vegetation",
    'TYPE_TREE_TRUNK': "trunk",
    'TYPE_CURB': "sidewalk",
    'TYPE_ROAD': "road",
    'TYPE_LANE_MARKER': "lane-marking",
    'TYPE_OTHER_GROUND': "other-ground",
    'TYPE_WALKABLE': "other-ground",
    'TYPE_SIDEWALK': "sidewalk"
}

waymo2kitti_safe= {
    'TYPE_UNDEFINED': "unlabeled",
    'TYPE_CAR': "car",
    'TYPE_TRUCK': "truck",
    'TYPE_BUS': "bus",
    'TYPE_OTHER_VEHICLE': "other-vehicle",
    'TYPE_MOTORCYCLIST': "motorcyclist",
    'TYPE_BICYCLIST': "bicyclist",
    'TYPE_PEDESTRIAN': "person",
    'TYPE_SIGN': "unlabeled",
    'TYPE_TRAFFIC_LIGHT': "unlabeled",
    'TYPE_POLE': "unlabeled",
    'TYPE_CONSTRUCTION_CONE': "unlabeled",
    'TYPE_BICYCLE': "bicycle",
    'TYPE_MOTORCYCLE': "motorcycle",
    'TYPE_BUILDING': "building",
    'TYPE_VEGETATION': "vegetation",
    'TYPE_TREE_TRUNK': "trunk",
    'TYPE_CURB': "unlabeled",
    'TYPE_ROAD': "unlabeled",
    'TYPE_LANE_MARKER': "lane-marking",
    'TYPE_OTHER_GROUND': "unlabeled",
    'TYPE_WALKABLE': "unlabeled",
    'TYPE_SIDEWALK': "sidewalk"
}

kitti2waymo = {
    "unlabeled": 'TYPE_UNDEFINED',                # "unlabeled"
    "outlier": 'TYPE_UNDEFINED',                  # "outlier" mapped to "unlabeled" ------------------ mapped
    "car": 'TYPE_CAR',                            # "car"
    "bicycle": 'TYPE_BICYCLE',                    # "bicycle"
    "bus": 'TYPE_BUS',                            # "bus" mapped to "other-vehicle" ------------------ mapped
    "motorcycle": 'TYPE_MOTORCYCLE',              # "motorcycle"
    "on-rails": 'TYPE_OTHER_VEHICLE',             # "on-rails" mapped to "other-vehicle" ------------- mapped
    "truck": 'TYPE_TRUCK',                        # "truck"
    "other-vehicle": 'TYPE_OTHER_VEHICLE',        # "other-vehicle"
    "person": 'TYPE_PEDESTRIAN',                  # "person"
    "bicyclist": 'TYPE_BICYCLIST',                # "bicyclist"
    "motorcyclist": 'TYPE_MOTORCYCLIST',          # "motorcyclist"
    "road": 'TYPE_ROAD',                          # "road"
    "parking": 'TYPE_ROAD',                       # "parking"
    "sidewalk": 'TYPE_SIDEWALK',                  # "sidewalk"
    "other-ground": 'TYPE_OTHER_GROUND',          # "other-ground"
    "building": 'TYPE_BUILDING',                  # "building"
    "fence": 'TYPE_UNDEFINED',                    # "fence"
    "other-structure": 'TYPE_UNDEFINED',          # "other-structure" mapped to "unlabeled" ---------- mapped
    "lane-marking": 'TYPE_LANE_MARKER',           # "lane-marking" to "road" ------------------------- mapped
    "vegetation": 'TYPE_VEGETATION',              # "vegetation"
    "trunk": 'TYPE_TREE_TRUNK',                   # "trunk"
    "terrain": 'TYPE_OTHER_GROUND',               # "terrain"
    "pole": 'TYPE_POLE',                          # "pole"
    "traffic-sign": 'TYPE_UNDEFINED',                  # "traffic-sign"
    "other-object": 'TYPE_UNDEFINED',             # "other-object" to "unlabeled" -------------------- mapped
    "moving-car": 'TYPE_CAR',                     # "moving-car" to "car" ---------------------------- mapped
    "moving-bicyclist": 'TYPE_BICYCLIST',         # "moving-bicyclist" to "bicyclist" ---------------- mapped
    "moving-person": 'TYPE_PEDESTRIAN',           # "moving-person" to "person" ---------------------- mapped
    "moving-motorcyclist": 'TYPE_MOTORCYCLIST',   # "moving-motorcyclist" to "motorcyclist" ---------- mapped
    "moving-on-rails": 'TYPE_OTHER_VEHICLE',      # "moving-on-rails" mapped to "other-vehicle" ------ mapped
    "moving-bus": 'TYPE_BUS',                     # "moving-bus" mapped to "other-vehicle" ----------- mapped
    "moving-truck": 'TYPE_TRUCK',                 # "moving-truck" to "truck" ------------------------ mapped
    "moving-other-vehicle": 'TYPE_OTHER_VEHICLE'          # "moving-other"-vehicle to "other-vehicle" -------- mapped
}

kitti2waymo_safe = {
    "unlabeled": 'TYPE_UNDEFINED',                # "unlabeled"
    "outlier": 'TYPE_UNDEFINED',                  # "outlier" mapped to "unlabeled" ------------------ mapped
    "car": 'TYPE_CAR',                            # "car"
    "bicycle": 'TYPE_BICYCLE',                    # "bicycle"
    "bus": 'TYPE_BUS',                            # "bus" mapped to "other-vehicle" ------------------ mapped
    "motorcycle": 'TYPE_MOTORCYCLE',              # "motorcycle"
    "on-rails": 'TYPE_OTHER_VEHICLE',             # "on-rails" mapped to "other-vehicle" ------------- mapped
    "truck": 'TYPE_TRUCK',                        # "truck"
    "other-vehicle": 'TYPE_OTHER_VEHICLE',        # "other-vehicle"
    "person": 'TYPE_PEDESTRIAN',                  # "person"
    "bicyclist": 'TYPE_BICYCLIST',                # "bicyclist"
    "motorcyclist": 'TYPE_MOTORCYCLIST',          # "motorcyclist"
    "road": 'TYPE_UNDEFINED',                          # "road"
    "parking": 'TYPE_UNDEFINED',                       # "parking"
    "sidewalk": 'TYPE_SIDEWALK',                  # "sidewalk"
    "other-ground": 'TYPE_UNDEFINED',          # "other-ground"
    "building": 'TYPE_BUILDING',                  # "building"
    "fence": 'TYPE_UNDEFINED',                    # "fence"
    "other-structure": 'TYPE_UNDEFINED',          # "other-structure" mapped to "unlabeled" ---------- mapped
    "lane-marking": 'TYPE_LANE_MARKER',           # "lane-marking" to "road" ------------------------- mapped
    "vegetation": 'TYPE_VEGETATION',              # "vegetation"
    "trunk": 'TYPE_TREE_TRUNK',                   # "trunk"
    "terrain": 'TYPE_UNDEFINED',               # "terrain"
    "pole": 'TYPE_UNDEFINED',                          # "pole"
    "traffic-sign": 'TYPE_UNDEFINED',                  # "traffic-sign"
    "other-object": 'TYPE_UNDEFINED',             # "other-object" to "unlabeled" -------------------- mapped
    "moving-car": 'TYPE_CAR',                     # "moving-car" to "car" ---------------------------- mapped
    "moving-bicyclist": 'TYPE_BICYCLIST',         # "moving-bicyclist" to "bicyclist" ---------------- mapped
    "moving-person": 'TYPE_PEDESTRIAN',           # "moving-person" to "person" ---------------------- mapped
    "moving-motorcyclist": 'TYPE_MOTORCYCLIST',   # "moving-motorcyclist" to "motorcyclist" ---------- mapped
    "moving-on-rails": 'TYPE_OTHER_VEHICLE',      # "moving-on-rails" mapped to "other-vehicle" ------ mapped
    "moving-bus": 'TYPE_BUS',                     # "moving-bus" mapped to "other-vehicle" ----------- mapped
    "moving-truck": 'TYPE_TRUCK',                 # "moving-truck" to "truck" ------------------------ mapped
    "moving-other-vehicle": 'TYPE_OTHER_VEHICLE'          # "moving-other"-vehicle to "other-vehicle" -------- mapped
}

kitti = {
    0 : "unlabeled",
    1 : "outlier" ,
    10: "car",
    11: "bicycle",
    13: "bus" ,
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure" ,
    60: "lane-marking" ,
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object" ,
    252: "moving-car", 
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails", 
    257: "moving-bus",
    258: "moving-truck" ,
    259: "moving-other-vehicle"
}

kitti_inv = {
    "unlabeled": 0,
    "outlier": 1,
    "car": 10,
    "bicycle": 11,
    "bus": 13,
    "motorcycle": 15,
    "on-rails": 16,
    "truck": 18,
    "other-vehicle": 20,
    "person": 30,
    "bicyclist": 31,
    "motorcyclist": 32,
    "road": 40,
    "parking": 44,
    "sidewalk": 48,
    "other-ground": 49,
    "building": 50,
    "fence": 51,
    "other-structure": 52,
    "lane-marking": 60,
    "vegetation": 70,
    "trunk": 71,
    "terrain": 72,
    "pole": 80,
    "traffic-sign": 81,
    "other-object": 99,
    "moving-car": 252,
    "moving-bicyclist": 253,
    "moving-person": 254,
    "moving-motorcyclist": 255,
    "moving-on-rails": 256,
    "moving-bus": 257,
    "moving-truck": 258,
    "moving-other-vehicle": 259
}


waymo = {
    0: 'TYPE_UNDEFINED',
    1: 'TYPE_CAR',
    2: 'TYPE_TRUCK',
    3: 'TYPE_BUS',
    4: 'TYPE_OTHER_VEHICLE',
    5: 'TYPE_MOTORCYCLIST',
    6: 'TYPE_BICYCLIST',
    7: 'TYPE_PEDESTRIAN',
    8: 'TYPE_SIGN',
    9: 'TYPE_TRAFFIC_LIGHT',
    10: 'TYPE_POLE',
    11: 'TYPE_CONSTRUCTION_CONE',
    12: 'TYPE_BICYCLE',
    13: 'TYPE_MOTORCYCLE',
    14: 'TYPE_BUILDING',
    15: 'TYPE_VEGETATION',
    16: 'TYPE_TREE_TRUNK',
    17: 'TYPE_CURB',
    18: 'TYPE_ROAD',
    19: 'TYPE_LANE_MARKER',
    20: 'TYPE_OTHER_GROUND',
    21: 'TYPE_WALKABLE',
    22: 'TYPE_SIDEWALK'
}

waymo_inv = {
    'TYPE_UNDEFINED': 0,
    'TYPE_CAR': 1,
    'TYPE_TRUCK': 2,
    'TYPE_BUS': 3,
    'TYPE_OTHER_VEHICLE': 4,
    'TYPE_MOTORCYCLIST': 5,
    'TYPE_BICYCLIST': 6,
    'TYPE_PEDESTRIAN': 7,
    'TYPE_SIGN': 8,
    'TYPE_TRAFFIC_LIGHT': 9,
    'TYPE_POLE': 10,
    'TYPE_CONSTRUCTION_CONE': 11,
    'TYPE_BICYCLE': 12,
    'TYPE_MOTORCYCLE': 13,
    'TYPE_BUILDING': 14,
    'TYPE_VEGETATION': 15,
    'TYPE_TREE_TRUNK': 16,
    'TYPE_CURB': 17,
    'TYPE_ROAD': 18,
    'TYPE_LANE_MARKER': 19,
    'TYPE_OTHER_GROUND': 20,
    'TYPE_WALKABLE': 21,
    'TYPE_SIDEWALK': 22
}

sem2sem = {
    0 : 0,     # "unlabeled"
    1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,     # "motorcycle"
    16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"
    31: 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5     # "moving-other-vehicle" to "other-vehicle" ----------------mapped
}


waymovalidfreqs = [160951307,
 69475897,
 7765250,
 3272170,
 1828020,
 503,
 248386,
 5913634,
 4865385,
 502838,
 8645906,
 488421,
 171792,
 197329,
 243989204,
 159686476,
 13575661,
 11267605,
 190620777,
 5444214,
 4198428,
 74503790,
 45916607]


w2k = {k: kitti_inv[waymo2kitti[waymo[k]]] for k in waymo.keys()}
sq_w2k = {k: sem2sem[w2k[k]] for k in w2k.keys()}

k2w = {k: waymo_inv[kitti2waymo[kitti[k]]] for k in kitti.keys()}
sq_k2w = k2w

w2k_safe = {k: kitti_inv[waymo2kitti_safe[waymo[k]]] for k in waymo.keys()}
sq_w2k_safe = {k: sem2sem[w2k_safe[k]] for k in w2k.keys()}

k2w_safe = {k: waymo_inv[kitti2waymo_safe[kitti[k]]] for k in kitti.keys()}
sq_k2w_safe = k2w_safe


