from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import ctypes
import numpy as np
import draw_performace

ll = ctypes.cdll.LoadLibrary
lib = ll("./init.so")
test_lib = ll("./test.so")
valid_lib = ll("./valid.so")
lib.setInPath("./data/FB15K/".encode('ascii'))
test_lib.setInPath("./data/FB15K/".encode('ascii'))
valid_lib.setInPath("./data/FB15K/".encode('ascii'))

class Config(object):
    def __init__(self):
        self.testFlag = True
        #self.testFlag = False
        self.hidden_size = 150
        self.nbatches = 500
        self.valid_size = 1000
        self.batch_size = 0
        self.entity_nums = 0
        self.relation_nums = 0
        self.trainTimes = 1500
        self.margin = 1.0
        self.no_cuda = False

class TransE(nn.Module):
    def __init__(self, config):
        super(TransE, self).__init__()
        entity_nums = config.entity_nums
        relation_nums = config.relation_nums
        entity_dim = config.hidden_size
        relation_dim = config.hidden_size
        self.entity_embeddings = nn.Embedding(entity_nums, entity_dim)
        self.relation_embeddings = nn.Embedding(relation_nums, relation_dim)
        nn.init.xavier_uniform(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform(self.relation_embeddings.weight.data)
        F.normalize(self.entity_embeddings.weight.data, p = 2)
        F.normalize(self.relation_embeddings.weight.data, p = 2)

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_h_e = self.entity_embeddings(pos_h)
        pos_r_e = self.relation_embeddings(pos_r)
        pos_t_e = self.entity_embeddings(pos_t)
        neg_h_e = self.entity_embeddings(neg_h)
        neg_r_e = self.relation_embeddings(neg_r)
        neg_t_e = self.entity_embeddings(neg_t)

        p_score = torch.abs(pos_h_e + pos_r_e - pos_t_e)
        n_score = torch.abs(neg_h_e + neg_r_e - neg_t_e)

        p_score = p_score.view(-1, 1, 150)
        n_score = n_score.view(-1, 1, 150)
        pos = torch.sum(torch.mean(p_score, 1), 1)
        neg = torch.sum(torch.mean(n_score, 1), 1)

        criterion = nn.MarginRankingLoss(1, False).cuda()
        y = Variable(torch.Tensor([-1])).cuda()
        loss = criterion(pos, neg, y)
        #loss = torch.sum(torch.max(p_score - n_score + config.margin, torch.tensor(0, dtype=torch.float32).cuda()))

        return loss

    def predict(self, h, r, t):

        ph_e = self.entity_embeddings(h)
        pr_e = self.relation_embeddings(r)
        pt_e = self.entity_embeddings(t)
        predict = torch.sum(torch.abs(ph_e + pr_e - pt_e), 1)
        return predict

#@draw_performace.track_plot
def train(model, optimizer, ph, pr, pt, nh, nr, nt, vph, vpr, vpt, vnh, vnr, vnt):
        train_loss = 0
        for batch in range(config.nbatches):
            lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)

            ph_t = torch.tensor(ph, dtype=torch.int64).cuda()
            pr_t = torch.tensor(pr, dtype=torch.int64).cuda()
            pt_t = torch.tensor(pt, dtype=torch.int64).cuda()
            nh_t = torch.tensor(nh, dtype=torch.int64).cuda()
            nr_t = torch.tensor(nr, dtype=torch.int64).cuda()
            nt_t = torch.tensor(nt, dtype=torch.int64).cuda()
            optimizer.zero_grad()
            loss = model(ph_t, pr_t, pt_t, nh_t, nr_t, nt_t)


            train_loss = train_loss + loss

            loss.backward()
            optimizer.step()

        test_loss = 0
        total = valid_lib.getTripleTotal()
        for batch in range(total/config.valid_size):
            valid_lib.getBatch(vph_addr, vpt_addr, vpr_addr, vnh_addr, vnt_addr, vnr_addr, config.batch_size)
            hp = torch.tensor(vph, dtype=torch.int64).cuda()
            rp = torch.tensor(vpr, dtype=torch.int64).cuda()
            tp = torch.tensor(vpt, dtype=torch.int64).cuda()
            hn = torch.tensor(vnh, dtype=torch.int64).cuda()
            rn = torch.tensor(vnr, dtype=torch.int64).cuda()
            tn = torch.tensor(vnt, dtype=torch.int64).cuda()
            #res = model(hp, rp, tp, hn, rn, tn)
            #test_loss = test_loss + res
        #test_loss = model.predict(h, r, t)
        train_loss = (train_loss/483142) * 50000
        test_loss = test_loss

        print('\nTrain set: Epoch Num: {:.4f} \t Average train_loss: {:.4f} \t Average test_loss: {:.4f}\n'.format(epoch, train_loss, test_loss))
        return train_loss, test_loss

def test(model, ph, pr, pt):
    total = test_lib.getTestTotal()
    for times in range(total):
        test_lib.getHeadBatch(ph_addr, pt_addr, pr_addr)

        h = torch.tensor(ph, dtype=torch.int64).cuda()
        r = torch.tensor(pr, dtype=torch.int64).cuda()
        t = torch.tensor(pt, dtype=torch.int64).cuda()
        res = model.predict(h, r, t)

        res_arr = res.data.cpu().numpy()
        test_lib.testHead(res_arr.__array_interface__['data'][0])

        test_lib.getTailBatch(ph_addr, pt_addr, pr_addr)
        h = torch.tensor(ph, dtype=torch.int64).cuda()
        r = torch.tensor(pr, dtype=torch.int64).cuda()
        t = torch.tensor(pt, dtype=torch.int64).cuda()
        loss = model.predict(h, r, t)
        loss_arr = loss.data.cpu().numpy()
        test_lib.testTail(loss_arr.__array_interface__['data'][0])
        print(times)
    test_lib.test()



if __name__ == "__main__":
    config = Config()
    if config.testFlag:
        test_lib.init()
        batch = test_lib.getEntityTotal()
        config.batch_size = batch
    else:
        lib.init()
        valid_lib.init()
        config.relation_nums = lib.getRelationTotal()
        config.entity_nums = lib.getEntityTotal()
        config.batch_size = lib.getTripleTotal() // config.nbatches

    print("hello")
    ph = np.zeros(config.batch_size, dtype=np.int32)
    pt = np.zeros(config.batch_size, dtype=np.int32)
    pr = np.zeros(config.batch_size, dtype=np.int32)
    nh = np.zeros(config.batch_size, dtype=np.int32)
    nt = np.zeros(config.batch_size, dtype=np.int32)
    nr = np.zeros(config.batch_size, dtype=np.int32)

    ph_addr = ph.__array_interface__['data'][0]
    pt_addr = pt.__array_interface__['data'][0]
    pr_addr = pr.__array_interface__['data'][0]
    nh_addr = nh.__array_interface__['data'][0]
    nt_addr = nt.__array_interface__['data'][0]
    nr_addr = nr.__array_interface__['data'][0]

    lib.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_int]
    test_lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    test_lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    test_lib.testHead.argtypes = [ctypes.c_void_p]
    test_lib.testTail.argtypes = [ctypes.c_void_p]

    vph = np.zeros(config.valid_size, dtype=np.int32)
    vpt = np.zeros(config.valid_size, dtype=np.int32)
    vpr = np.zeros(config.valid_size, dtype=np.int32)
    vnh = np.zeros(config.valid_size, dtype=np.int32)
    vnt = np.zeros(config.valid_size, dtype=np.int32)
    vnr = np.zeros(config.valid_size, dtype=np.int32)
    vph_addr = vph.__array_interface__['data'][0]
    vpt_addr = vpt.__array_interface__['data'][0]
    vpr_addr = vpr.__array_interface__['data'][0]
    vnh_addr = vnh.__array_interface__['data'][0]
    vnt_addr = vnt.__array_interface__['data'][0]
    vnr_addr = vnr.__array_interface__['data'][0]
    valid_lib.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_int]

    #use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda")

    if not config.testFlag:
        model = TransE(config).to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        for epoch in range(config.trainTimes):
            model.train()
            train_loss, test_loss = train(model, optimizer, ph, pr, pt, nh, nr, nt, vph, vpr, vpt, vnh, vnr, vnt)

        torch.save(model, 'transE.pkl')
        print(model.state_dict())
    else:
        model = torch.load('transE.pkl')
        model.eval()
        test(model, ph, pr, pt)









