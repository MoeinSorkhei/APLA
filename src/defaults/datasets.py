from torchvision import datasets as torch_datasets
from pathlib import Path

from utils import *
from .bases import BaseSet


def get_vtab_data_loc(dataset_name):
    if dataset_name in ['VTAB_flowers', '_VTAB_flowers']:
        return 'VTAB_oxford_flowers102'
    if dataset_name == 'VTAB_pets':
        return 'VTAB_oxford_iiit_pet'
    if dataset_name == 'VTAB_pcam':
        return 'VTAB_patch_camelyon'
    if dataset_name == 'VTAB_clevr_count':
        return 'VTAB_clevr_count_all'
    if dataset_name == 'VTAB_clevr_dist':
        return 'VTAB_clevr_closest_object_distance'
    if dataset_name == 'VTAB_dsprites_loc':
        return 'VTAB_dsprites_label_x_position'
    if dataset_name == 'VTAB_dsprites_ori':
        return 'VTAB_dsprites_label_orientation'
    if dataset_name == 'VTAB_smallnorb_azimuth':
        return 'VTAB_smallnorb_label_azimuth'
    if dataset_name == 'VTAB_smallnorb_elevation':
        return 'VTAB_smallnorb_label_elevation'
    if dataset_name == 'VTAB_kitti_dist':
        return 'VTAB_kitti_closest_vehicle_distance'
    if dataset_name == 'VTAB_retinopathy':
        return 'VTAB_diabetic_retinopathy_detection'
    if dataset_name == 'VTAB_svhn':
        return 'VTAB_svhn_cropped'
    return dataset_name  # same name as dataset


class VTABDataset(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'accuracy'
    mean = (0.485, 0.456, 0.406)  # imagenet
    std = (0.229, 0.224, 0.225)  # imagenet

    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = get_vtab_data_loc(self.__class__.__name__)
        print(f'--- VTAB dataset_location for {self.__class__.__name__}: {self.dataset_location}')
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()

    def get_data_as_list(self):
        if hasattr(self, 'train_val') and self.train_val:
            print(f'LOADING TRAIN_VAL MODE: {self.mode}')
            if self.mode == 'train':
                path1 = os.path.join(self.root_dir, 'train')
                path2 = os.path.join(self.root_dir, 'val')
                files1 = helpfuns.files_with_suffix(path1, '.png')
                files2 = helpfuns.files_with_suffix(path2, '.png')
                files = files1 + files2
            else:
                path = os.path.join(self.root_dir, 'test')
                files = helpfuns.files_with_suffix(path, '.png')
            print(f'Len files: {len(files)}')
        else:
            print(f'LOADING REGULARLY MODE: {self.mode}')
            path = os.path.join(self.root_dir, self.mode)
            files = helpfuns.files_with_suffix(path, '.png')
        
        # exclude pets corrupted image
        files = [filepath for filepath in files if not filepath.endswith('VTAB_oxford_iiit_pet/train/img_261-label_20.png')]
        files = [filepath for filepath in files if not filepath.endswith('VTAB_sun397/train/img_442-label_85.png')]
        data_list = [{
            'img_path': f,
            'label': int(f.split('.')[0].split('-label_')[1])
        } for f in files]
        #
        n_classes = len(set([item['label'] for item in data_list]))
        print(f'--- VTAB dataset {self.__class__.__name__} {self.mode} set has len: {len(data_list):,} -- {n_classes} classes\n')
        #
        return data_list


class VTAB_cifar100(VTABDataset):
    n_classes = 100
    n_images = 11_000  # train: 800, val: 200, test: 10,000


class VTAB_caltech101(VTABDataset):
    n_classes = 102
    n_images = 7_084  # train/val/test: 800 / 200 / 6,084


class VTAB_dtd(VTABDataset):
    n_classes = 47
    # 800 / 200 / 1,880


class VTAB_flowers(VTABDataset):
    n_classes = 102
    # 800 / 200 / 6,149


class _VTAB_flowers(VTABDataset):
    n_classes = 102
    # 800 / 200 / 6,149


class VTAB_pets(VTABDataset):
    n_classes = 37
    # 800 / 200 / 3,669


class VTAB_svhn(VTABDataset):
    n_classes = 10
    # 800 / 200 / 26,032


class VTAB_sun397(VTABDataset):
    n_classes = 397
    # 1,241 / 200 / 21,750


class VTAB_pcam(VTABDataset):
    n_classes = 2
    # 800 / 200 / 32,768


class VTAB_eurosat(VTABDataset):
    n_classes = 10
    # 800 / 200 / 5,400


class VTAB_resisc45(VTABDataset):
    n_classes = 45
    # 800 / 200 / 6,300


class VTAB_retinopathy(VTABDataset):
    n_classes = 5
    # 931 / 200 / 42,670


class VTAB_clevr_count(VTABDataset):
    n_classes = 8
    # train / val / test: 1,506 / 200 / 15,000


class VTAB_clevr_dist(VTABDataset):
    n_classes = 6
    # 800 / 200 / 15,000


class VTAB_dmlab(VTABDataset):
    n_classes = 6
    # 800 / 200 / 22,735


class VTAB_kitti_dist(VTABDataset):
    n_classes = 4
    # 1,225 / 200 / 711


class VTAB_dsprites_loc(VTABDataset):
    n_classes = 16
    # 800 / 200 / 73,728


class VTAB_dsprites_ori(VTABDataset):
    n_classes = 16
    # 800 / 200 / 73,728


class VTAB_smallnorb_azimuth(VTABDataset):
    n_classes = 18
    # 1,554 / 200 / 12,150


class VTAB_smallnorb_elevation(VTABDataset):
    n_classes = 9
    # 1,521 / 200 / 12,150


class NABirds(BaseSet):
    # ref: https://dl.allaboutbirds.org/nabirds and https://gvanhorn38.github.io/assets/papers/building_a_bird_recognition_app.pdf
    # notes: test split and train_val splits were provided, val_split was constructed from 0.3 of shuffled train_val
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'accuracy'
    mean = (0.492, 0.508, 0.464)
    std = (0.218, 0.217, 0.264)
    int_to_labels = {
        0: '295',  # int -> class_id, to see the actual name of each category see leaf_classes.txt
        1: '296',
        2: '297',
        3: '298',
        4: '299',
        5: '313',
        6: '314',
        7: '315',
        8: '316',
        9: '317',
        10: '318',
        11: '319',
        12: '320',
        13: '321',
        14: '322',
        15: '323',
        16: '324',
        17: '325',
        18: '326',
        19: '327',
        20: '328',
        21: '329',
        22: '330',
        23: '331',
        24: '332',
        25: '333',
        26: '334',
        27: '335',
        28: '336',
        29: '337',
        30: '338',
        31: '339',
        32: '340',
        33: '341',
        34: '342',
        35: '343',
        36: '344',
        37: '345',
        38: '346',
        39: '347',
        40: '348',
        41: '349',
        42: '350',
        43: '351',
        44: '352',
        45: '353',
        46: '354',
        47: '355',
        48: '356',
        49: '357',
        50: '358',
        51: '359',
        52: '360',
        53: '361',
        54: '362',
        55: '363',
        56: '364',
        57: '365',
        58: '366',
        59: '367',
        60: '368',
        61: '369',
        62: '370',
        63: '371',
        64: '372',
        65: '373',
        66: '374',
        67: '375',
        68: '376',
        69: '377',
        70: '378',
        71: '379',
        72: '380',
        73: '381',
        74: '382',
        75: '392',
        76: '393',
        77: '394',
        78: '395',
        79: '396',
        80: '397',
        81: '398',
        82: '399',
        83: '400',
        84: '401',
        85: '402',
        86: '446',
        87: '447',
        88: '448',
        89: '449',
        90: '450',
        91: '451',
        92: '452',
        93: '453',
        94: '454',
        95: '455',
        96: '456',
        97: '457',
        98: '458',
        99: '459',
        100: '460',
        101: '461',
        102: '462',
        103: '463',
        104: '464',
        105: '465',
        106: '466',
        107: '467',
        108: '468',
        109: '469',
        110: '470',
        111: '471',
        112: '472',
        113: '473',
        114: '474',
        115: '475',
        116: '476',
        117: '477',
        118: '478',
        119: '479',
        120: '480',
        121: '481',
        122: '482',
        123: '483',
        124: '484',
        125: '485',
        126: '486',
        127: '487',
        128: '488',
        129: '489',
        130: '490',
        131: '491',
        132: '492',
        133: '493',
        134: '494',
        135: '495',
        136: '496',
        137: '497',
        138: '498',
        139: '499',
        140: '500',
        141: '501',
        142: '502',
        143: '503',
        144: '504',
        145: '505',
        146: '506',
        147: '507',
        148: '508',
        149: '509',
        150: '510',
        151: '511',
        152: '512',
        153: '513',
        154: '514',
        155: '515',
        156: '516',
        157: '517',
        158: '518',
        159: '519',
        160: '520',
        161: '521',
        162: '522',
        163: '523',
        164: '524',
        165: '525',
        166: '526',
        167: '527',
        168: '528',
        169: '529',
        170: '530',
        171: '531',
        172: '532',
        173: '533',
        174: '534',
        175: '535',
        176: '536',
        177: '537',
        178: '538',
        179: '539',
        180: '540',
        181: '541',
        182: '542',
        183: '543',
        184: '544',
        185: '545',
        186: '546',
        187: '547',
        188: '548',
        189: '549',
        190: '550',
        191: '551',
        192: '552',
        193: '553',
        194: '554',
        195: '555',
        196: '556',
        197: '557',
        198: '558',
        199: '559',
        200: '560',
        201: '561',
        202: '599',
        203: '600',
        204: '601',
        205: '602',
        206: '603',
        207: '604',
        208: '605',
        209: '606',
        210: '607',
        211: '608',
        212: '609',
        213: '610',
        214: '611',
        215: '612',
        216: '613',
        217: '614',
        218: '615',
        219: '616',
        220: '617',
        221: '618',
        222: '619',
        223: '620',
        224: '621',
        225: '622',
        226: '623',
        227: '624',
        228: '625',
        229: '626',
        230: '627',
        231: '628',
        232: '629',
        233: '630',
        234: '631',
        235: '632',
        236: '633',
        237: '634',
        238: '635',
        239: '636',
        240: '637',
        241: '638',
        242: '639',
        243: '640',
        244: '641',
        245: '642',
        246: '643',
        247: '644',
        248: '645',
        249: '646',
        250: '647',
        251: '648',
        252: '649',
        253: '650',
        254: '651',
        255: '652',
        256: '653',
        257: '654',
        258: '655',
        259: '656',
        260: '657',
        261: '658',
        262: '659',
        263: '660',
        264: '661',
        265: '662',
        266: '663',
        267: '664',
        268: '665',
        269: '666',
        270: '667',
        271: '668',
        272: '669',
        273: '670',
        274: '671',
        275: '672',
        276: '673',
        277: '674',
        278: '675',
        279: '676',
        280: '677',
        281: '678',
        282: '679',
        283: '680',
        284: '681',
        285: '696',
        286: '697',
        287: '698',
        288: '699',
        289: '700',
        290: '746',
        291: '747',
        292: '748',
        293: '749',
        294: '750',
        295: '751',
        296: '752',
        297: '753',
        298: '754',
        299: '755',
        300: '756',
        301: '757',
        302: '758',
        303: '759',
        304: '760',
        305: '761',
        306: '762',
        307: '763',
        308: '764',
        309: '765',
        310: '766',
        311: '767',
        312: '768',
        313: '769',
        314: '770',
        315: '771',
        316: '772',
        317: '773',
        318: '774',
        319: '775',
        320: '776',
        321: '777',
        322: '778',
        323: '779',
        324: '780',
        325: '781',
        326: '782',
        327: '783',
        328: '784',
        329: '785',
        330: '786',
        331: '787',
        332: '788',
        333: '789',
        334: '790',
        335: '791',
        336: '792',
        337: '793',
        338: '794',
        339: '795',
        340: '796',
        341: '797',
        342: '798',
        343: '799',
        344: '800',
        345: '801',
        346: '802',
        347: '803',
        348: '804',
        349: '805',
        350: '806',
        351: '807',
        352: '808',
        353: '809',
        354: '810',
        355: '811',
        356: '812',
        357: '813',
        358: '814',
        359: '815',
        360: '816',
        361: '817',
        362: '818',
        363: '819',
        364: '820',
        365: '821',
        366: '822',
        367: '823',
        368: '824',
        369: '825',
        370: '826',
        371: '827',
        372: '828',
        373: '829',
        374: '830',
        375: '831',
        376: '832',
        377: '833',
        378: '834',
        379: '835',
        380: '836',
        381: '837',
        382: '838',
        383: '839',
        384: '840',
        385: '841',
        386: '842',
        387: '843',
        388: '844',
        389: '845',
        390: '846',
        391: '847',
        392: '848',
        393: '849',
        394: '850',
        395: '851',
        396: '852',
        397: '853',
        398: '854',
        399: '855',
        400: '856',
        401: '857',
        402: '858',
        403: '859',
        404: '860',
        405: '861',
        406: '862',
        407: '863',
        408: '864',
        409: '865',
        410: '866',
        411: '867',
        412: '868',
        413: '869',
        414: '870',
        415: '871',
        416: '872',
        417: '873',
        418: '874',
        419: '875',
        420: '876',
        421: '877',
        422: '878',
        423: '879',
        424: '880',
        425: '881',
        426: '882',
        427: '883',
        428: '884',
        429: '885',
        430: '886',
        431: '887',
        432: '888',
        433: '889',
        434: '890',
        435: '891',
        436: '892',
        437: '893',
        438: '894',
        439: '895',
        440: '896',
        441: '897',
        442: '898',
        443: '899',
        444: '900',
        445: '901',
        446: '902',
        447: '903',
        448: '904',
        449: '905',
        450: '906',
        451: '907',
        452: '908',
        453: '909',
        454: '910',
        455: '911',
        456: '912',
        457: '913',
        458: '914',
        459: '915',
        460: '916',
        461: '917',
        462: '918',
        463: '919',
        464: '920',
        465: '921',
        466: '922',
        467: '923',
        468: '924',
        469: '925',
        470: '926',
        471: '927',
        472: '928',
        473: '929',
        474: '930',
        475: '931',
        476: '932',
        477: '933',
        478: '934',
        479: '935',
        480: '936',
        481: '937',
        482: '938',
        483: '939',
        484: '940',
        485: '941',
        486: '942',
        487: '943',
        488: '944',
        489: '945',
        490: '946',
        491: '947',
        492: '948',
        493: '949',
        494: '950',
        495: '951',
        496: '952',
        497: '953',
        498: '954',
        499: '955',
        500: '956',
        501: '957',
        502: '958',
        503: '959',
        504: '960',
        505: '961',
        506: '962',
        507: '963',
        508: '964',
        509: '965',
        510: '966',
        511: '967',
        512: '968',
        513: '969',
        514: '970',
        515: '971',
        516: '972',
        517: '973',
        518: '974',
        519: '975',
        520: '976',
        521: '977',
        522: '978',
        523: '979',
        524: '980',
        525: '981',
        526: '982',
        527: '983',
        528: '984',
        529: '985',
        530: '986',
        531: '987',
        532: '988',
        533: '989',
        534: '990',
        535: '991',
        536: '992',
        537: '993',
        538: '994',
        539: '995',
        540: '996',
        541: '997',
        542: '998',
        543: '999',
        544: '1000',
        545: '1001',
        546: '1002',
        547: '1003',
        548: '1004',
        549: '1005',
        550: '1006',
        551: '1007',
        552: '1008',
        553: '1009',
        554: '1010'
    }
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    n_classes = len(int_to_labels)
    n_images = 48_562

    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        data_df = pd.read_csv(os.path.join(self.root_dir, 'data_info.csv'))  # data_infor has records for all the data
        if self.mode == 'all':
            selected_df = data_df
        else:
            if hasattr(self, 'train_val'):
                print(f'LOADING TRAIN_VAL with mode: {self.mode}')
                if self.mode == 'train':
                    selected_image_ids = []
                    for file in ['train_image_ids.txt', 'val_image_ids.txt']:
                        with open(os.path.join(self.root_dir, file), 'r') as f:
                            selected = f.read().split('\n')[:-1]  # last line empty    
                            selected_image_ids.extend(selected)
                            # print(f'extended with len: {len(selected)}')
                            # input()
                else:
                    file = 'test_image_ids.txt'  # val and test
                    with open(os.path.join(self.root_dir, file), 'r') as f:
                        selected_image_ids = f.read().split('\n')[:-1]  # last line empty  
            else:
                print_ddp(f'LOADING REGULARLY with mode: {self.mode}')
                file = 'train_image_ids.txt' if self.mode == 'train' else 'val_image_ids.txt' if self.mode == 'val' else 'test_image_ids.txt'
                with open(os.path.join(self.root_dir, file), 'r') as f:
                    selected_image_ids = f.read().split('\n')[:-1]  # last line empty    
            selected_df = data_df[data_df['image_id'].isin(selected_image_ids)]

        data_list = [{
            'img_path': os.path.join(self.root_dir, 'images', row['imagepath']),
            'label': self.labels_to_int[str(row['class_id'])]
        } for _, row in selected_df.iterrows()]
        return data_list


class DDSM(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'   
    knn_nhood = 200
    target_metric = 'roc_auc'
    n_classes = 2
    
    def __init__(self, dataset_params, mode='train', n_class=2, is_patch=False):
        self.attr_from_dict(dataset_params)
        self.mode = mode
        self.n_class = n_class
        self.is_patch = is_patch
        self.export_labels_as_int()
        self.init_stats()
        self.n_classes = len(self.int_to_labels)
        assert self.n_classes == self.n_class
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.label_mode = '{}class'.format(self.n_class)
        
        self.data = self.get_dataset()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self, data_loc):
        data_list = []
        data = pd.read_csv(data_loc, sep=" ", header=None, engine='python')
        if self.is_patch:
            data.columns = ["img_path", "label"]
            for img_path, label in zip(data['img_path'], data['label']):
                img_path = os.path.join(*img_path.split("/")[1:])
                img_path = os.path.join(self.root_dir, 'ddsm_patches', img_path)
                data_list.append({'img_path': img_path, 'label': label, 'dataset': self.name})
        else:
            data.columns = ["img_path"]
            txt_to_lbl = {'normal': 0, 'benign': 1, 'cancer': 2}
            for img_path in data['img_path']:
                img_path = os.path.join(self.root_dir, 'ddsm_raw', img_path)
                label = os.path.basename(img_path).split("_")[0]
                label = txt_to_lbl[label]
                if self.n_classes == 2 and label > 1:
                    label = 1
                if not self.is_multiclass:
                    label = [float(label)]
                data_list.append({'img_path': img_path, 'label': label, 'dataset': self.name})
                    
        return data_list
    
    def get_dataset(self):
        if self.is_patch:
            self.df_path = os.path.join(self.root_dir, 'ddsm_labels', self.label_mode)
        else:
            self.df_path = os.path.join(self.root_dir, 'ddsm_raw_image_lists')
        if self.mode == 'train':
            self.df_path = os.path.join(self.df_path, 'train.txt')
        elif self.mode in ['val', 'eval']:
            self.df_path = os.path.join(self.df_path, 'val.txt')
        elif self.mode == 'test':
            self.df_path = os.path.join(self.df_path, 'test.txt')
        return self.get_data_as_list(self.df_path)
            
    def init_stats(self):
        if self.is_patch:
            self.mean = (0.44,) * self.img_channels
            self.std = (0.25,) * self.img_channels
        else:
            self.mean = (0.286,) * self.img_channels
            self.std = (0.267,) * self.img_channels
        
    def export_labels_as_int(self):
        if self.n_class == 3:
            self.int_to_labels = {
                0: 'Normal',
                1: 'Benign',
                2: 'Cancer'
            }
        else:
            self.int_to_labels = {
                0: 'Normal',
                1: 'Cancer'
            }
        self.labels_to_int = {val: key for key, val in self.int_to_labels.items()} 


class ISIC2019(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    mean = [0.66776717, 0.52960888, 0.52434725]
    std = [0.22381877, 0.20363036, 0.21538623]
    knn_nhood = 200    
    int_to_labels = {
        0: 'Melanoma',
        1: 'Melanocytic nevus',
        2: 'Basal cell carcinoma',
        3: 'Actinic keratosis',
        4: 'Benign keratosis',
        5: 'Dermatofibroma',
        6: 'Vascular lesion',
        7: 'Squamous cell carcinoma'
    }
    target_metric = 'recall'
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        data_list = []
        datainfo = pd.read_csv(os.path.join(self.root_dir, 'ISIC_2019_Training_GroundTruth.csv'), engine='python')
        metadata = pd.read_csv(os.path.join(self.root_dir, 'ISIC_2019_Training_Metadata.csv'), engine='python')
        labellist = datainfo.values[:, 1:].nonzero()[1].tolist()
        img_names = datainfo.values[:, 0].tolist()
        img_names = [os.path.join(self.root_dir, 'train',  imname + '.jpg') for imname in img_names]
        dataframe = pd.DataFrame(list(zip(img_names, labellist)), 
                                 columns=['img_path', 'label'])
        
        val_id_json = os.path.join(self.root_dir, 'val_ids.json')
        train_ids, test_val_ids = self.get_validation_ids(total_size=len(dataframe), val_size=0.2, 
                                                          json_path=val_id_json, 
                                                          dataset_name=self.name)
        val_ids = test_val_ids[:int(len(test_val_ids)/2)]
        test_ids = test_val_ids[int(len(test_val_ids)/2):]     
        
        if hasattr(self, 'train_val'):
            print(f'DOING TRAIN_VAL')
            if self.mode == 'train':
                data = dataframe.loc[train_ids + val_ids, :]
            else:
                data = dataframe.loc[test_ids, :]
        else:
            if self.mode == 'train':
                data = dataframe.loc[train_ids, :]
            elif self.mode in ['val', 'eval']:
                data = dataframe.loc[val_ids, :]
            else:
                data = dataframe.loc[test_ids, :]
        labels = data['label'].values.tolist()
        img_paths = data['img_path'].values.tolist()
        data_list = [{'img_path': img_path, 'label': label, 'dataset': self.name}
                     for img_path, label in zip(img_paths, labels)]
                    
        return data_list  
    
    
class APTOS2019(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    mean = (0.415, 0.221, 0.073)
    std = (0.275, 0.150, 0.081)
    int_to_labels = {
        0: 'No DR',
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        4: 'Proliferative DR'
    }
    target_metric = 'quadratic_kappa'
    knn_nhood = 200    
    n_classes = len(int_to_labels)
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        data_list = []
        datainfo = pd.read_csv(os.path.join(self.root_dir, 'train.csv'), engine='python')
        labellist = datainfo.diagnosis.tolist()
        img_names = datainfo.id_code.tolist()
        img_names = [os.path.join(self.root_dir, 'train_images', imname + '.png') for imname in img_names]
        dataframe = pd.DataFrame(list(zip(img_names, labellist)), 
                                 columns=['img_path', 'label'])
        
        val_id_json = os.path.join(self.root_dir, 'val_ids.json')
        train_ids, test_val_ids = self.get_validation_ids(total_size=len(dataframe), val_size=0.3, 
                                                          json_path=val_id_json, 
                                                          dataset_name=self.name)
        val_ids = test_val_ids[:int(len(test_val_ids)/2)]
        test_ids = test_val_ids[int(len(test_val_ids)/2):]     
        
        if self.mode == 'train':
            data = dataframe.loc[train_ids, :]
        elif self.mode in ['val', 'eval']:
            data = dataframe.loc[val_ids, :]
        else:
            data = dataframe.loc[test_ids, :]
        labels = data['label'].values.tolist()
        img_paths = data['img_path'].values.tolist()
        data_list = [{'img_path': img_path, 'label': label, 'dataset': self.name}
                     for img_path, label in zip(img_paths, labels)]
                    
        return data_list    


class Flowers102(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    mean = (0.435, 0.38, 0.292)
    std = (0.293, 0.243, 0.27)
    int_to_labels = {
        0: 'pink primrose',
        1: 'hard-leaved pocket orchid',
        2: 'canterbury bells',
        3: 'sweet pea',
        4: 'english marigold',
        5: 'tiger lily',
        6: 'moon orchid',
        7: 'bird of paradise',
        8: 'monkshood',
        9: 'globe thistle',
        10: 'snapdragon',
        11: "colt's foot",
        12: 'king protea',
        13: 'spear thistle',
        14: 'yellow iris',
        15: 'globe-flower',
        16: 'purple coneflower',
        17: 'peruvian lily',
        18: 'balloon flower',
        19: 'giant white arum lily',
        20: 'fire lily',
        21: 'pincushion flower',
        22: 'fritillary',
        23: 'red ginger',
        24: 'grape hyacinth',
        25: 'corn poppy',
        26: 'prince of wales feathers',
        27: 'stemless gentian',
        28: 'artichoke',
        29: 'sweet william',
        30: 'carnation',
        31: 'garden phlox',
        32: 'love in the mist',
        33: 'mexican aster',
        34: 'alpine sea holly',
        35: 'ruby-lipped cattleya',
        36: 'cape flower',
        37: 'great masterwort',
        38: 'siam tulip',
        39: 'lenten rose',
        40: 'barbeton daisy',
        41: 'daffodil',
        42: 'sword lily',
        43: 'poinsettia',
        44: 'bolero deep blue',
        45: 'wallflower',
        46: 'marigold',
        47: 'buttercup',
        48: 'oxeye daisy',
        49: 'common dandelion',
        50: 'petunia',
        51: 'wild pansy',
        52: 'primula',
        53: 'sunflower',
        54: 'pelargonium',
        55: 'bishop of llandaff',
        56: 'gaura',
        57: 'geranium',
        58: 'orange dahlia',
        59: 'pink-yellow dahlia',
        60: 'cautleya spicata',
        61: 'japanese anemone',
        62: 'black-eyed susan',
        63: 'silverbush',
        64: 'californian poppy',
        65: 'osteospermum',
        66: 'spring crocus',
        67: 'bearded iris',
        68: 'windflower',
        69: 'tree poppy',
        70: 'gazania',
        71: 'azalea',
        72: 'water lily',
        73: 'rose',
        74: 'thorn apple',
        75: 'morning glory',
        76: 'passion flower',
        77: 'lotus',
        78: 'toad lily',
        79: 'anthurium',
        80: 'frangipani',
        81: 'clematis',
        82: 'hibiscus',
        83: 'columbine',
        84: 'desert-rose',
        85: 'tree mallow',
        86: 'magnolia',
        87: 'cyclamen',
        88: 'watercress',
        89: 'canna lily',
        90: 'hippeastrum',
        91: 'bee balm',
        92: 'ball moss',
        93: 'foxglove',
        94: 'bougainvillea',
        95: 'camellia',
        96: 'mallow',
        97: 'mexican petunia',
        98: 'bromelia',
        99: 'blanket flower',
        100: 'trumpet creeper',
        101: 'blackberry lily'
    }
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    n_classes = len(int_to_labels)
    n_images = 8189
    target_metric = 'mean_per_class_accuracy'
        
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        csv_file = 'train.csv' if self.mode == 'train' else 'val.csv' if self.mode == 'val' else 'test.csv' if self.mode == 'test' else 'all_labels.csv'
        csv_path = os.path.join(self.root_dir, csv_file)
        df = pd.read_csv(csv_path)

        data_list = [{
            'img_path': os.path.join(self.root_dir, 'images', row['filename']),
            'label': int(row['label']) - 1  # because their labels start from 1 instead of 0
            } for _, row in df.iterrows()
        ]
        return data_list


class SUN397(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'accuracy'
    mean = (0.473, 0.456, 0.42)
    std = (0.258, 0.256, 0.279)
    n_classes = 397
    # there are multiple paritions for train/test data, and we take the first one, following https://arxiv.org/abs/1805.08974 (Appendix A.2)
    n_images = 39_700
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        sun397 = torch_datasets.SUN397(root=self.root_dir, download=False)
        
        train_val_imgfiles = helpfuns.read_file_to_list(os.path.join(self.root_dir, 'Partitions', 'Training_01.txt'))
        test_imgfiles = helpfuns.read_file_to_list(os.path.join(self.root_dir, 'Partitions', 'Testing_01.txt'))  # first partition
        val_imgfiles = helpfuns.read_file_to_list(os.path.join(self.root_dir, 'val_imagefiles.txt'))  # training: 45 images per category, val: 5 images per categ (uniformly sampled already)
        
        train_imagefiles = [tr_imfile for tr_imfile in train_val_imgfiles if tr_imfile not in val_imgfiles]  # exclude image files from train set that are in val set
        
        if self.mode == 'train':
            selected_imgfiles = train_imagefiles
        elif self.mode == 'val':
            selected_imgfiles = val_imgfiles
        elif self.mode == 'test':
            selected_imgfiles = test_imgfiles
        elif self.mode == 'all':
            selected_imgfiles = train_imagefiles + val_imgfiles + test_imgfiles
        else:
            raise NotImplementedError
        
        sun397_copy = [str(path) for path in sun397._image_files]
        selected_inds = [sun397_copy.index(sun397.root + '/SUN397' + im_file) for im_file in selected_imgfiles]   # use the indexes to iterate over data
        # print(f'len selected_inds: {len(selected_inds)}')
        
        data_list = [{
            'img_path': str(sun397._image_files[ind]),  # convert posix path to str
            'label': sun397._labels[ind]
        } for ind in selected_inds]
        
        return data_list


class CIFAR_10(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'accuracy'
    mean = (0.493, 0.484, 0.448)
    std = (0.241, 0.237, 0.256)
    n_classes = 10
    n_images = 60_000
    
    train_data = None
    val_data = None
    test_data = None
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        if self.train_data is None:
            self.train_data, self.val_data, self.test_data = self.retrieve_data(which_cifar=self.__class__.__name__, 
                                                                                root_dir=self.root_dir)
            print(f'Initialized our train/val/test splits')
        
        if self.mode == 'train':
            selected_data = self.train_data
        elif self.mode == 'val':
            selected_data = self.val_data
        elif self.mode == 'test':
            selected_data = self.test_data
        elif self.mode == 'all':
            selected_data = {
                'images': np.concatenate((self.train_data['images'], self.val_data['images'], self.test_data['images']), axis=0),  # e.g. shape (40000, 32, 32, 3)
                'labels': np.concatenate((self.train_data['labels'], self.val_data['labels'], self.test_data['labels']), axis=0)
            }
        else:
            raise NotImplementedError
        
        data_list = [{
            'img_arr': selected_data['images'][i],
            'label': selected_data['labels'][i]
        } for i in range(len(selected_data['images']))]
        return data_list
    
    @staticmethod
    def retrieve_data(which_cifar, root_dir):
        print(f'Retrieving data for: {which_cifar}')
        if which_cifar == 'CIFAR_10':
            cifar_class = torch_datasets.CIFAR10  
        elif which_cifar == 'CIFAR_100':
            cifar_class = torch_datasets.CIFAR100  
        else:
            raise NotImplementedError
        
        cifar_train = cifar_class(root=root_dir,
                                  train=True,
                                  download=False)
        cifar_test = cifar_class(root=root_dir,
                                 train=False,
                                 download=False)
        train_data = {
            'images': cifar_train.data[:40000],  # images as np array
            'labels':cifar_train.targets[:40000]
        }
        val_data = {
            'images': cifar_train.data[40000:],
            'labels': cifar_train.targets[40000:]
        }
        test_data = {
            'images': cifar_test.data,
            'labels': cifar_test.targets
        }
        return train_data, val_data, test_data


class CIFAR_100(CIFAR_10):
    # everything else implemented in the parent class
    mean = (0.508, 0.487, 0.441)
    std = (0.263, 0.252, 0.272)
    n_classes = 100
    n_images = 60_000


class Colorectal(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'accuracy'   # page 4 of: https://www.nature.com/articles/srep27988 (they used 10-fold cross-validation)
    mean = (0.654, 0.475, 0.586)
    std = (0.252, 0.325, 0.266)
    n_classes = 8
    n_images = 5_000
    classes = [
        '01_TUMOR',
        '02_STROMA',
        '03_COMPLEX',
        '04_LYMPHO',
        '05_DEBRIS',
        '06_MUCOSA',
        '07_ADIPOSE',
        '08_EMPTY'
    ]
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        # csv_file = os.path.join(self.root_dir, f'{self.mode}.csv')
        files_path = os.path.join(self.root_dir, f'{self.mode}.txt')
        paths = helpfuns.read_file_to_list(files_path)
        # data_df = pd.read_csv(csv_file)
        data_list = [{
            'img_path': os.path.join(self.root_dir, path),  # reisezd images for faster processing
            'label': self.classes.index(path.split('/')[-2])
        } for path in paths]
        return data_list


class AID(BaseSet):
    # reference: https://captain-whu.github.io/AID/
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'accuracy'
    mean = (0.401, 0.413, 0.372)
    std = (0.21, 0.187, 0.185)
    int_to_labels = {
        0: 'Airport',
        1: 'BareLand',
        2: 'BaseballField',
        3: 'Beach',
        4: 'Bridge',
        5: 'Center',
        6: 'Church',
        7: 'Commercial',
        8: 'DenseResidential',
        9: 'Desert',
        10: 'Farmland',
        11: 'Forest',
        12: 'Industrial',
        13: 'Meadow',
        14: 'MediumResidential',
        15: 'Mountain',
        16: 'Park',
        17: 'Parking',
        18: 'Playground',
        19: 'Pond',
        20: 'Port',
        21: 'RailwayStation',
        22: 'Resort',
        23: 'River',
        24: 'School',
        25: 'SparseResidential',
        26: 'Square',
        27: 'Stadium',
        28: 'StorageTanks',
        29: 'Viaduct'
    }
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    n_classes = len(int_to_labels)
    n_images = 10_000

    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()

    def get_data_as_list(self):
        all_filepaths = helpfuns.files_with_suffix(os.path.join(self.root_dir, 'images'), suffix='.jpg')
        if hasattr(self, 'train_val'):
            print(f'DOING TRAIN_VAL')
            if self.mode == 'train':
                df_train = pd.read_csv(os.path.join(self.root_dir, 'train.csv'))
                df_val = pd.read_csv(os.path.join(self.root_dir, 'val.csv'))
                df = pd.concat([df_train, df_val], ignore_index=True)
            else:
                df = pd.read_csv(os.path.join(self.root_dir, 'test.csv'))
        else:
            csv_file = 'train.csv' if self.mode == 'train' else 'val.csv' if self.mode == 'val' else 'test.csv' if self.mode == 'test' else 'all_labels.csv'
            df = pd.read_csv(os.path.join(self.root_dir, csv_file))
        selected_filepaths = [filepath for filepath in all_filepaths if os.path.split(filepath)[-1] in df['filename'].tolist()]

        data_list = [{
            'img_path': filepath,
            'label': self.labels_to_int[os.path.split(Path(filepath).parent.absolute())[1]]
            } for filepath in selected_filepaths]
        return data_list


class RSSCN7(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'mean_per_class_accuracy'
    mean = (0.402, 0.409, 0.38)   # (0.388, 0.397, 0.362),
    std = (0.2, 0.18, 0.183)   # (0.198, 0.177, 0.179)
    int_to_labels = {
        0: 'aGrass',
        1: 'bField',
        2: 'cIndustry',
        3: 'dRiverLake',
        4: 'eForest',
        5: 'fResident',
        6: 'gParking'
    }
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    n_classes = len(int_to_labels)
    n_images = 2_800
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()

    def _foldername_from_file(self, filename):
        if filename.startswith('a'):
            return 'aGrass'
        elif filename.startswith('b'):
            return 'bField'
        elif filename.startswith('c'):
            return 'cIndustry'
        elif filename.startswith('d'):
            return 'dRiverLake'
        elif filename.startswith('e'):
            return 'eForest'
        elif filename.startswith('f'):
            return 'fResident'
        elif filename.startswith('g'):
            return 'gParking'
        return NotImplementedError

    def get_data_as_list(self):
        csv_file = 'train.csv' if self.mode == 'train' else 'val.csv' if self.mode == 'val' else 'test.csv' if self.mode == 'test' else 'all_labels.csv'
        csv_path = os.path.join(self.root_dir, csv_file)
        df = pd.read_csv(csv_path)

        data_list = [{
            'img_path': os.path.join(self.root_dir, 'images', self._foldername_from_file(row['filename']), row['filename']),
            'label': int(row['label'])
            } for _, row in df.iterrows()
        ]
        return data_list


class Aircraft(BaseSet):
    # ref: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'mean_per_class_accuracy'
    mean = (0.478, 0.509, 0.533)
    std = (0.217, 0.21, 0.242)
    int_to_labels = {  # 100 variants
        0: '707-320',
        1: '727-200',
        2: '737-200',
        3: '737-300',
        4: '737-400',
        5: '737-500',
        6: '737-600',
        7: '737-700',
        8: '737-800',
        9: '737-900',
        10: '747-100',
        11: '747-200',
        12: '747-300',
        13: '747-400',
        14: '757-200',
        15: '757-300',
        16: '767-200',
        17: '767-300',
        18: '767-400',
        19: '777-200',
        20: '777-300',
        21: 'A300B4',
        22: 'A310',
        23: 'A318',
        24: 'A319',
        25: 'A320',
        26: 'A321',
        27: 'A330-200',
        28: 'A330-300',
        29: 'A340-200',
        30: 'A340-300',
        31: 'A340-500',
        32: 'A340-600',
        33: 'A380',
        34: 'ATR-42',
        35: 'ATR-72',
        36: 'An-12',
        37: 'BAE 146-200',
        38: 'BAE 146-300',
        39: 'BAE-125',
        40: 'Beechcraft 1900',
        41: 'Boeing 717',
        42: 'C-130',
        43: 'C-47',
        44: 'CRJ-200',
        45: 'CRJ-700',
        46: 'CRJ-900',
        47: 'Cessna 172',
        48: 'Cessna 208',
        49: 'Cessna 525',
        50: 'Cessna 560',
        51: 'Challenger 600',
        52: 'DC-10',
        53: 'DC-3',
        54: 'DC-6',
        55: 'DC-8',
        56: 'DC-9-30',
        57: 'DH-82',
        58: 'DHC-1',
        59: 'DHC-6',
        60: 'DHC-8-100',
        61: 'DHC-8-300',
        62: 'DR-400',
        63: 'Dornier 328',
        64: 'E-170',
        65: 'E-190',
        66: 'E-195',
        67: 'EMB-120',
        68: 'ERJ 135',
        69: 'ERJ 145',
        70: 'Embraer Legacy 600',
        71: 'Eurofighter Typhoon',
        72: 'F-16A/B',
        73: 'F/A-18',
        74: 'Falcon 2000',
        75: 'Falcon 900',
        76: 'Fokker 100',
        77: 'Fokker 50',
        78: 'Fokker 70',
        79: 'Global Express',
        80: 'Gulfstream IV',
        81: 'Gulfstream V',
        82: 'Hawk T1',
        83: 'Il-76',
        84: 'L-1011',
        85: 'MD-11',
        86: 'MD-80',
        87: 'MD-87',
        88: 'MD-90',
        89: 'Metroliner',
        90: 'Model B200',
        91: 'PA-28',
        92: 'SR-20',
        93: 'Saab 2000',
        94: 'Saab 340',
        95: 'Spitfire',
        96: 'Tornado',
        97: 'Tu-134',
        98: 'Tu-154',
        99: 'Yak-42'
    }
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    n_classes = len(int_to_labels)
    n_images = 10_200

    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    # maybe todo for later: 
    # comment this function and refactor so it makes a simple call to get_data_as_list_for_level()
    def get_data_as_list(self):
        textfile = 'images_variant_train.txt' if self.mode == 'train' else 'images_variant_val.txt' if self.mode == 'val' \
            else 'images_variant_test.txt'  if self.mode == 'test' else 'images_variant_all.txt'
        with open (os.path.join(self.root_dir, 'data', textfile), 'r') as file:
            lines = file.read().split('\n')[:-1]  # last line empty

        data_list = [{
            'img_path': os.path.join(self.root_dir, 'data', 'images', f"{line[:7]}.jpg"),
            'label': self.labels_to_int[str(line[8:])]
        } for line in lines]
        return data_list
    
    def get_data_as_list_for_level(self, level):
        textfile = f'images_{level}_train.txt' if self.mode == 'train' else f'images_{level}_val.txt' if self.mode == 'val' \
            else f'images_{level}_test.txt'  if self.mode == 'test' else f'images_{level}_all.txt'
        with open (os.path.join(self.root_dir, 'data', textfile), 'r') as file:
            lines = file.read().split('\n')[:-1]  # last line empty

        data_list = [{
            'img_path': os.path.join(self.root_dir, 'data', 'images', f"{line[:7]}.jpg"),
            'label': self.labels_to_int[str(line[8:])]
        } for line in lines]
        return data_list


class StanfordCars(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'  # ref: page 7 of https://ai.stanford.edu/~jkrause/papers/3drr13.pdf
    knn_nhood = 200
    target_metric = 'accuracy'
    mean = (0.469, 0.459, 0.454)
    std = (0.29, 0.289, 0.297)
    n_classes = 196
    n_images = 16_185
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
    
    def open_image(self, img_path):
        return Image.open(img_path).convert("RGB")  # changes nothing, just to be consistent with torch
    
    def get_data_as_list(self):
        train_data = torch_datasets.StanfordCars(root=self.root_dir, split='train', download=False)
        test_data = torch_datasets.StanfordCars(root=self.root_dir, split='test', download=False)
        
        val_imgfiles = helpfuns.read_file_to_list(os.path.join(self.root_dir, 'val_imgfiles.txt'))
        val_filepaths = [os.path.join(self.root_dir, path) for path in val_imgfiles]
        
        if hasattr(self, 'train_val'):
            print(f'DOING TRAIN_VAL')
            if self.mode == 'train':
                selected_data = [sample for sample in train_data._samples]
            else:
                selected_data = test_data._samples  # train and val
        else:
            if self.mode == 'train':
                selected_data = [sample for sample in train_data._samples if sample[0] not in val_filepaths]  # _samples is a list of tuples (filepath, target)
            elif self.mode == 'val':
                selected_data = [sample for sample in train_data._samples if sample[0] in val_filepaths]
            elif self.mode == 'test':
                selected_data = test_data._samples
            elif self.mode == 'all':
                selected_data = train_data._samples + test_data._samples
            else:
                raise NotImplementedError
        
        print(f'mode: {self.mode}, len selected_data: {len(selected_data)}')
        
        data_list = [{
            'img_path': sample[0],
            'label': sample[1]
        } for sample in selected_data]
        
        return data_list


class DTD(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'accuracy'
    mean = (0.531, 0.474, 0.425)
    std = (0.265, 0.255, 0.263)
    n_classes = 47
    n_images = 5_640
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
    
    def get_data_as_list(self):
        dtd_train = torch_datasets.DTD(root=self.root_dir, split='train', download=False)  # partition may be used for stuff like cross-validation
        dtd_val = torch_datasets.DTD(root=self.root_dir, split='val', download=False)
        dtd_test = torch_datasets.DTD(root=self.root_dir, split='test', download=False)
        
        if self.mode == 'train':
            imagepaths = dtd_train._image_files
            labels = dtd_train._labels
            
        elif self.mode == 'val':
            imagepaths = dtd_val._image_files
            labels = dtd_val._labels
        
        elif self.mode == 'test':
            imagepaths = dtd_test._image_files
            labels = dtd_test._labels
        
        elif self.mode == 'all':
            imagepaths = dtd_train._image_files + dtd_val._image_files + dtd_test._image_files
            labels = dtd_train._labels + dtd_val._labels + dtd_test._labels
        else:
            raise NotImplementedError
        
        data_list = [{
            'img_path': str(imagepaths[i]),
            'label': labels[i]
        } for i in range(len(imagepaths))]
        return data_list


class StanfordDogs(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'accuracy'  # top 1
    mean = (0.476, 0.452, 0.391)
    std = (0.259, 0.253, 0.258)
    int_to_labels = {
        0: 'n02085620-Chihuahua',
        1: 'n02085782-Japanese_spaniel',
        2: 'n02085936-Maltese_dog',
        3: 'n02086079-Pekinese',
        4: 'n02086240-Shih-Tzu',
        5: 'n02086646-Blenheim_spaniel',
        6: 'n02086910-papillon',
        7: 'n02087046-toy_terrier',
        8: 'n02087394-Rhodesian_ridgeback',
        9: 'n02088094-Afghan_hound',
        10: 'n02088238-basset',
        11: 'n02088364-beagle',
        12: 'n02088466-bloodhound',
        13: 'n02088632-bluetick',
        14: 'n02089078-black-and-tan_coonhound',
        15: 'n02089867-Walker_hound',
        16: 'n02089973-English_foxhound',
        17: 'n02090379-redbone',
        18: 'n02090622-borzoi',
        19: 'n02090721-Irish_wolfhound',
        20: 'n02091032-Italian_greyhound',
        21: 'n02091134-whippet',
        22: 'n02091244-Ibizan_hound',
        23: 'n02091467-Norwegian_elkhound',
        24: 'n02091635-otterhound',
        25: 'n02091831-Saluki',
        26: 'n02092002-Scottish_deerhound',
        27: 'n02092339-Weimaraner',
        28: 'n02093256-Staffordshire_bullterrier',
        29: 'n02093428-American_Staffordshire_terrier',
        30: 'n02093647-Bedlington_terrier',
        31: 'n02093754-Border_terrier',
        32: 'n02093859-Kerry_blue_terrier',
        33: 'n02093991-Irish_terrier',
        34: 'n02094114-Norfolk_terrier',
        35: 'n02094258-Norwich_terrier',
        36: 'n02094433-Yorkshire_terrier',
        37: 'n02095314-wire-haired_fox_terrier',
        38: 'n02095570-Lakeland_terrier',
        39: 'n02095889-Sealyham_terrier',
        40: 'n02096051-Airedale',
        41: 'n02096177-cairn',
        42: 'n02096294-Australian_terrier',
        43: 'n02096437-Dandie_Dinmont',
        44: 'n02096585-Boston_bull',
        45: 'n02097047-miniature_schnauzer',
        46: 'n02097130-giant_schnauzer',
        47: 'n02097209-standard_schnauzer',
        48: 'n02097298-Scotch_terrier',
        49: 'n02097474-Tibetan_terrier',
        50: 'n02097658-silky_terrier',
        51: 'n02098105-soft-coated_wheaten_terrier',
        52: 'n02098286-West_Highland_white_terrier',
        53: 'n02098413-Lhasa',
        54: 'n02099267-flat-coated_retriever',
        55: 'n02099429-curly-coated_retriever',
        56: 'n02099601-golden_retriever',
        57: 'n02099712-Labrador_retriever',
        58: 'n02099849-Chesapeake_Bay_retriever',
        59: 'n02100236-German_short-haired_pointer',
        60: 'n02100583-vizsla',
        61: 'n02100735-English_setter',
        62: 'n02100877-Irish_setter',
        63: 'n02101006-Gordon_setter',
        64: 'n02101388-Brittany_spaniel',
        65: 'n02101556-clumber',
        66: 'n02102040-English_springer',
        67: 'n02102177-Welsh_springer_spaniel',
        68: 'n02102318-cocker_spaniel',
        69: 'n02102480-Sussex_spaniel',
        70: 'n02102973-Irish_water_spaniel',
        71: 'n02104029-kuvasz',
        72: 'n02104365-schipperke',
        73: 'n02105056-groenendael',
        74: 'n02105162-malinois',
        75: 'n02105251-briard',
        76: 'n02105412-kelpie',
        77: 'n02105505-komondor',
        78: 'n02105641-Old_English_sheepdog',
        79: 'n02105855-Shetland_sheepdog',
        80: 'n02106030-collie',
        81: 'n02106166-Border_collie',
        82: 'n02106382-Bouvier_des_Flandres',
        83: 'n02106550-Rottweiler',
        84: 'n02106662-German_shepherd',
        85: 'n02107142-Doberman',
        86: 'n02107312-miniature_pinscher',
        87: 'n02107574-Greater_Swiss_Mountain_dog',
        88: 'n02107683-Bernese_mountain_dog',
        89: 'n02107908-Appenzeller',
        90: 'n02108000-EntleBucher',
        91: 'n02108089-boxer',
        92: 'n02108422-bull_mastiff',
        93: 'n02108551-Tibetan_mastiff',
        94: 'n02108915-French_bulldog',
        95: 'n02109047-Great_Dane',
        96: 'n02109525-Saint_Bernard',
        97: 'n02109961-Eskimo_dog',
        98: 'n02110063-malamute',
        99: 'n02110185-Siberian_husky',
        100: 'n02110627-affenpinscher',
        101: 'n02110806-basenji',
        102: 'n02110958-pug',
        103: 'n02111129-Leonberg',
        104: 'n02111277-Newfoundland',
        105: 'n02111500-Great_Pyrenees',
        106: 'n02111889-Samoyed',
        107: 'n02112018-Pomeranian',
        108: 'n02112137-chow',
        109: 'n02112350-keeshond',
        110: 'n02112706-Brabancon_griffon',
        111: 'n02113023-Pembroke',
        112: 'n02113186-Cardigan',
        113: 'n02113624-toy_poodle',
        114: 'n02113712-miniature_poodle',
        115: 'n02113799-standard_poodle',
        116: 'n02113978-Mexican_hairless',
        117: 'n02115641-dingo',
        118: 'n02115913-dhole',
        119: 'n02116738-African_hunting_dog'
    }
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    n_classes = len(int_to_labels)
    n_images = 20_580

    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
    
    def get_data_as_list(self):
        csv_file = 'train.csv' if self.mode == 'train' else 'val.csv' if self.mode == 'val' else 'test.csv' if self.mode == 'test' else 'all_labels.csv'
        df = pd.read_csv(os.path.join(self.root_dir, csv_file))

        data_list = [{
            'img_path': os.path.join(self.root_dir, 'images', row['filename']),  # filename is like: n02116738-African_hunting_dog/n02116738_6330.jpg
            'label': int(row['label'])
            } for _, row in df.iterrows()
        ]
        return data_list


class OxfordIII_Pet(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'mean_per_class_accuracy'  # ref: page 17 of https://arxiv.org/abs/2007.08489
    mean = (0.482, 0.449, 0.395)
    std = (0.265, 0.26, 0.268)
    n_classes = 37
    n_images = 7_349
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        oxf_pets_trainval = torch_datasets.OxfordIIITPet(root=self.root_dir, 
                                                         split='trainval', 
                                                         target_types='category', 
                                                         transform=None, 
                                                         target_transform=None, 
                                                         download=False)
        oxf_pets_test = torch_datasets.OxfordIIITPet(root=self.root_dir, 
                                                     split='test', 
                                                     target_types='category',
                                                     transform=None, 
                                                     target_transform=None, 
                                                     download=False)

        val_imgfiles = helpfuns.read_file_to_list(os.path.join(self.root_dir, 'val_imgfiles.txt'))
        val_inds = [list(map(str, oxf_pets_trainval._images)).index(os.path.join(self.root_dir, imgfile)) for imgfile in val_imgfiles]  # gets inds of the val images
        train_inds = [ind for ind in range(len(oxf_pets_trainval._images)) if ind not in val_inds]  # get the inds of images that are not in val inds
        
        if self.mode == 'train':
            imagepaths = [oxf_pets_trainval._images[i] for i in train_inds]
            labels = [oxf_pets_trainval._labels[i] for i in train_inds]
        
        elif self.mode == 'val':
            imagepaths = [oxf_pets_trainval._images[i] for i in val_inds]
            labels = [oxf_pets_trainval._labels[i] for i in val_inds]
        
        elif self.mode == 'test':
            imagepaths = oxf_pets_test._images
            labels = oxf_pets_test._labels
        
        elif self.mode == 'all':
            imagepaths = oxf_pets_trainval._images + oxf_pets_test._images
            labels = oxf_pets_trainval._labels + oxf_pets_test._labels
        else:
            raise NotImplementedError

        data_list = [{
            'img_path': str(imagepaths[i]),
            'label': labels[i]
        } for i in range(len(imagepaths))]
        return data_list


class CUB_200_2011(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'accuracy'
    mean = (0.486, 0.5, 0.43)
    std = (0.228, 0.223, 0.262)
    n_classes = 200
    n_images = 11_788
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
    
    def get_data_as_list(self):
        csv_file = os.path.join(self.root_dir, f'{self.mode}.csv')
        data_df = pd.read_csv(csv_file)
        data_list = [{
            'img_path': os.path.join(self.root_dir, 'images', row['img_name']),
            'label': int(row['label'])
        } for _, row in data_df.iterrows()]
        return data_list


class Birdsnap(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'accuracy'
    mean = (0.488, 0.502, 0.456)
    std = (0.224, 0.221, 0.262)
    n_classes = 500
    n_images = 39_820  # not including the images with downloade failed
    # also skipping truncated file: Mourning_Dove/181010.jpg
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
    
    def get_data_as_list(self):
        csv_file = os.path.join(self.root_dir, f'{self.mode}.csv')
        data_df = pd.read_csv(csv_file)
        data_list = [{
            # 'img_path': os.path.join(self.root_dir, 'download', 'images', row['filepath']),
            'img_path': os.path.join(self.root_dir, 'download', 'images_256', row['filepath']),  # reisezd images for faster processing
            'label': int(row['label'])
        } for _, row in data_df.iterrows()]
        return data_list


class Caltech_101(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'mean_per_class_accuracy'  # ref (also info about splits): page 17 of https://arxiv.org/abs/2007.08489
    mean = (0.547, 0.526, 0.495)
    std = (0.32, 0.316, 0.327)
    n_classes = 101
    n_images = 8_677

    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
        
    def open_image(self, img_path):
        return Image.open(img_path).convert("RGB")  # changes nothing, just to be consistent with torch

    def get_data_as_list(self):
        if self.mode in ['train', 'val', 'test']:
            data_file = os.path.join(self.root_dir, f'{self.mode}.csv')
            data_df = pd.read_csv(data_file)
        elif self.mode == 'all':
            df_list = []
            for split in ['train', 'val', 'test']:
                df_list.append(pd.read_csv(os.path.join(self.root_dir, f'{split}.csv')))
            data_df = pd.concat(df_list, ignore_index=True)
        else:
            raise NotImplementedError
        
        data_list = [{
            'img_path': os.path.join(self.root_dir, row['filename']),
            'label': int(row['label'])
        } for _, row in data_df.iterrows()]  
        return data_list


class Caltech_256(Caltech_101):
    # most of the attributes inherited from the parent class
    stats = {
        'mean': (0.554, 0.536, 0.507),
        'std': (0.314, 0.312, 0.327)
    }
    n_classes = 257  # yes, that is correct :)
    n_images = 30_607


class MIT_Indoor(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'mean_per_class_accuracy'
    mean = (0.487, 0.43, 0.372)
    std = (0.263, 0.257, 0.259)
    n_classes = 67
    n_images = 6_700
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
    
    def get_data_as_list(self):
        csv_file = os.path.join(self.root_dir, f'{self.mode}.csv')
        data_df = pd.read_csv(csv_file)
        data_list = [{
            'img_path': os.path.join(self.root_dir, 'indoorCVPR_09', 'Images', row['filename']),  # reisezd images for faster processing
            'label': int(row['label'])
        } for _, row in data_df.iterrows()]
        return data_list


class Pneumonia(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    # multuple metrics are used according to page 7 of: https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5, 
    # but ROC AUC should be more interesting than accuracy
    target_metric = 'roc_auc'
    mean = (0.482, 0.482, 0.482)
    std = (0.236, 0.236, 0.236)
    n_classes = 2
    n_images = 5_856
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()

    def get_data_as_list(self):
        csv_file = os.path.join(self.root_dir, f'{self.mode}.csv')
        data_df = pd.read_csv(csv_file)
        data_list = [{
            'img_path': os.path.join(self.root_dir, 'images', row['filename']),  # reisezd images for faster processing
            'label': int(row['label'])
        } for _, row in data_df.iterrows()]
        return data_list


class ImageNet(BaseSet):
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    knn_nhood = 200
    target_metric = 'accuracy'
    mean = (0.485, 0.456, 0.406)  # imagenet
    std = (0.229, 0.224, 0.225)  # imagenet
    n_classes = 1_000
    # train: 1,281,167 -- val: 50,000
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = self.__class__.__name__
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        self.mode = mode
        
        self.imagenet_train = torch_datasets.ImageNet(root=self.root_dir, split='train', transform=None, target_transform=None)
        self.imagenet_val = torch_datasets.ImageNet(root=self.root_dir, split='val', transform=None, target_transform=None)
        self.data = self.get_data_as_list()
        
        self.transform, self.resizing = self.get_transforms()
        
    # def get_data_as_list(self):
    #     if self.mode == 'train':
    #         root_dir = os.path.join(self.root_dir, 'train')
    #     else:  # val + test
    #         root_dir = os.path.join(self.root_dir, 'val')
    #     images = helpfuns.files_with_suffix(root_dir, '.JPEG')
    #     # data_df = pd.read_csv(root_dir)
    #     data_list = [{
    #         'img_path': path,
    #         'label': os.path.split(helpfuns.get_parent_path(path))[-1]
    #     } for path in images]
    #     print(f'Number of {self.mode} image: {len(data_list):,}')
    #     return data_list
    
    def get_data_as_list(self):
        if self.mode == 'train':
            the_imagenet = self.imagenet_train
        else:  # val + test
            the_imagenet = self.imagenet_val
        
        samples = the_imagenet.samples
        data_list = [{
            'img_path': s[0],
            'label': s[1]
        } for s in samples]
        print_ddp(f'\nNumber of {self.mode} image: {len(data_list):,}')
        return data_list
