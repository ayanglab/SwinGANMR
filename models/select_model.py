'''
# -----------------------------------------
Define Training Model
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

def define_Model(opt):
    model = opt['model']

    if model == 'swinmr_stgan':
        from models.model_swinmr_stgan import MRI_STGAN as M

    elif model == 'swinmr_eesgan':
        from models.model_swinmr_eesgan import MRI_EESGAN as M

    elif model == 'swinmr_tesgan':
        from models.model_swinmr_tesgan import MRI_TESGAN as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
