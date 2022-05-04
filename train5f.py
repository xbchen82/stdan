from torch.utils.data import DataLoader
import loader2 as lo
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import os
from evaluate5f import Evaluate
from config import *


def maskedNLL(y_pred, y_gt, mask):
    # mask = t.cat((mask[:15, :, :], t.zeros_like(mask[15:, :, :])), dim=0)
    acc = t.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]
    sigY = y_pred[:, :, 3]
    rho = y_pred[:, :, 4]
    ohr = t.pow(1 - t.pow(rho, 2), -0.5)  # (1-rhp^2)^0.5
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    # If we represent likelihood in feet^(-1)
    out = 0.5 * t.pow(ohr, 2) * (
            t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY, 2) - 2 * rho * t.pow(sigX,
                                                                                                      1) * t.pow(
        sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379
    # If we represent likelihood in m^(-1):meter out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX,
    # 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX)
    # * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = t.sum(acc) / t.sum(mask)
    return lossVal




def MSELoss2(g_out, fut, mask):
    acc = t.zeros_like(mask)
    muX = g_out[:, :, 0]
    muY = g_out[:, :, 1]
    x = fut[:, :, 0]
    y = fut[:, :, 1]
    out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = t.sum(acc) / t.sum(mask)
    return lossVal


def CELoss(pred, target):
    value = t.log(t.sum(pred * target, dim=-1))
    return -t.sum(value) / value.shape[0]


def main():
    args['train_flag'] = True
    evaluate = Evaluate()
    #   test 
    # t1 = lo.NgsimDataset('data/5feature/TrainSet.mat')
    # t1.collate_fn([t1.__getitem__(4587)])
    gdEncoder = model.GDEncoder(args)
    generator = model.Generator(args)
    # generator.load_state_dict(t.load('checkpoints/4fx/epoch0_g.tar'))
    # discriminator.load_state_dict(t.load('checkpoints/4fx/epoch0_d.tar'))
    gdEncoder = gdEncoder.to(device)
    generator = generator.to(device)
    gdEncoder.train()
    generator.train()
    if dataset == "ngsim":
        if args['lon_length'] == 3:
            t1 = lo.NgsimDataset('../data/dataset_t_v_t/TrainSet.mat')
        else:
            t1 = lo.NgsimDataset('../data/5feature/TrainSet.mat')
        trainDataloader = DataLoader(t1, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                     collate_fn=t1.collate_fn) 
    else:
        t1 = lo.HighdDataset('../data/highD/TrainSet_highd.mat')
        trainDataloader = DataLoader(t1, batch_size=args['batch_size'], shuffle=True,
                                     collate_fn=t1.collate_fn)  
    optimizer_gd = optim.Adam(gdEncoder.parameters(), lr=learning_rate)
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    scheduler_gd = ExponentialLR(optimizer_gd, gamma=0.6)
    scheduler_g = ExponentialLR(optimizer_g, gamma=0.6)
    for epoch in range(args['epoch']):
        print("epoch:", epoch + 1, 'lr', optimizer_g.param_groups[0]['lr'])
        loss_gi1 = 0
        loss_gix = 0
        loss_gx_2i = 0
        loss_gx_3i = 0
        for idx, data in enumerate(tqdm(trainDataloader)):
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions = data
            hist = hist.to(device)
            nbrs = nbrs.to(device)
            mask = mask.to(device)
            lat_enc = lat_enc.to(device)
            lon_enc = lon_enc.to(device)
            fut = fut[:args['out_length'], :, :]
            fut = fut.to(device)
            op_mask = op_mask[:args['out_length'], :, :]
            op_mask = op_mask.to(device)
            va = va.to(device)
            nbrsva = nbrsva.to(device)
            lane = lane.to(device)
            nbrslane = nbrslane.to(device)
            dis = dis.to(device)
            nbrsdis = nbrsdis.to(device)
            map_positions = map_positions.to(device)
            cls = cls.to(device)
            nbrscls = nbrscls.to(device)
            values = gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
            g_out, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)
            if args['use_mse']:
                loss_g1 = MSELoss2(g_out, fut, op_mask)
            else:
                if epoch < args['pre_epoch']:
                    loss_g1 = MSELoss2(g_out, fut, op_mask)
                else:
                    loss_g1 = maskedNLL(g_out, fut, op_mask)
            loss_gx_3 = CELoss(lat_pred, lat_enc)
            loss_gx_2 = CELoss(lon_pred, lon_enc)
            loss_gx = loss_gx_3 + loss_gx_2
            loss_g = loss_g1 + 1 * loss_gx
            optimizer_g.zero_grad()
            optimizer_gd.zero_grad()
            loss_g.backward()
            a = t.nn.utils.clip_grad_norm_(generator.parameters(), 10)
            a = t.nn.utils.clip_grad_norm_(gdEncoder.parameters(), 10)
            optimizer_g.step()
            optimizer_gd.step()

            loss_gi1 += loss_g1.item()
            loss_gx_2i += loss_gx_2.item()
            loss_gx_3i += loss_gx_3.item()
            loss_gix += loss_gx.item()
            if idx % 10000 == 9999:
                print('mse:', loss_gi1 / 10000, '|loss_gx_2:', loss_gx_2i / 10000, '|loss_gx_3', loss_gx_3i / 10000)
                loss_gi1 = 0
                loss_gix = 0
                loss_gx_2i = 0
                loss_gx_3i = 0
            # if idx == int(len(trainDataloader) / 4) * model_step:
            #     print('mse:', loss_gi1 / int(len(trainDataloader) / 4), '|c1:',
            #           loss_gix / int(len(trainDataloader) / 4), '|c:', loss_gi3 / int(len(trainDataloader) / 4), '|d:',
            #           loss_gi4 / int(len(trainDataloader) / 4))
            #     loss_gi1 = 0
            #     loss_gix = 0
            #     loss_gi3 = 0
            #     loss_gi4 = 0
            #     model_step += 1
            #     if model_step == 4:
            #         model_step = 1
            #     if epoch >= 4:
            #         evaluate(name=str(epoch) + str(model_step), valDataloader=valDataloader, device=device,
            #                  cdgEncoder=cgdEncoder,
            #                  generator=generator, discriminator=discriminator, classified=classified,
            #                  f_length=args['f_length'], use_maneuvers=args['use_maneuvers'])

        save_model(name=str(epoch + 1), gdEncoder=gdEncoder,
                   generator=generator, path = args['path'])
        evaluate.main(name=str(epoch + 1), val=True)
        scheduler_gd.step()
        scheduler_g.step()


def save_model(name, gdEncoder, generator, path):
    l_path = args['path']
    if not os.path.exists(l_path):
        os.makedirs(l_path)
    t.save(gdEncoder.state_dict(), l_path + '/epoch' + name + '_gd.tar')
    t.save(generator.state_dict(), l_path + '/epoch' + name + '_g.tar')


if __name__ == '__main__':
    main()
