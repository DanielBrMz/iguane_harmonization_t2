IMAGE_SHAPE=[160,192,160,1]
# Fetal brain specific settings
FETAL_IMAGE_SHAPE = (138, 176, 1)  # From brain age model
FETAL_STACK_SHAPE = (138, 176, 4)  # 4 slices as channels
FETAL_NUM_SLICES = 4
FETAL_SITES = ['BCH_CHD', 'BCH_Placenta', 'dHCP', 'HBCD', 'TMC', 'VGH']
FETAL_GA_RANGE = (20, 40)  # Gestational age range in weeks