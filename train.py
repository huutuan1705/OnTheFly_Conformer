import torch
import torch.nn as nn
import torch.utils.data as data 
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from data.datasets import FGSBIR_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    
    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))
    
    return dataloader_train, dataloader_test

def evaluate_model(model, dataloader_test):
    with torch.no_grad():
        model.eval()
        Image_Feature_ALL = []
        Image_Name = []
        Sketch_Feature_ALL = []
        Sketch_Name = []
        for _, sampled_batch in enumerate(tqdm(dataloader_test, dynamic_ncols=False)):
            sketch_feature, positive_feature= model.test_forward(sampled_batch)
            Sketch_Feature_ALL.extend(sketch_feature)
            Sketch_Name.extend(sampled_batch['sketch_path'])

            for i_num, positive_name in enumerate(sampled_batch['positive_path']):
                if positive_name not in Image_Name:
                    Image_Name.append(sampled_batch['positive_path'][i_num])
                    Image_Feature_ALL.append(positive_feature[i_num])

        rank = torch.zeros(len(Sketch_Name))
        rank_percentile = torch.zeros(len(Sketch_Name))
        Image_Feature_ALL = torch.stack(Image_Feature_ALL)
        
        avererage_area = []
        avererage_area_percentile = []

        for num, sketch_feature in enumerate(Sketch_Feature_ALL):
            s_name = Sketch_Name[num]
            sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
            position_query = Image_Name.index(sketch_query_name)

            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                  Image_Feature_ALL[position_query].unsqueeze(0))

            rank[num] = distance.le(target_distance).sum()
            rank_percentile[num] = (len(distance) - rank[num]) / (len(distance) - 1)
            
            avererage_area.append(1/rank[num].item() if rank[num].item()!=0 else 1)
            avererage_area_percentile.append(rank_percentile[num].item() if rank_percentile[num].item!=0 else 1)

        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top5 = rank.le(5).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]
        
        return top1, top5, top10
    
def train_model(model, args):
    dataloader_train, dataloader_test = get_dataloader(args)
    
    loss_fn = nn.TripletMarginLoss(margin=args.margin)
    optimizer = optim.Adam([
            {'params': model.sample_embedding_network.parameters(), 'lr': args.lr},
            {'params': model.conformer.parameters(), 'lr': args.lr},
        ])
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    
    top1, top5, top10 = 0, 0, 0
    for i_epoch in range(args.epochs):
        print(f"Epoch: {i_epoch+1} / {args.epochs}")
        
        losses = []
        for _, batch_data in enumerate(tqdm(dataloader_train, dynamic_ncols=False)):
            model.train()
            optimizer.zero_grad()
            
            sketch_feature, positive_feature, negative_feature = model(batch_data)
            loss = loss_fn(sketch_feature, positive_feature, negative_feature)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            
            losses.append(loss.item())
        
        avg_loss = sum(losses) / len(losses)
        top1_eval, top5_eval, top10_eval = evaluate_model(model, dataloader_test)
        
        if top5_eval > top5:
            top1, top5, top10 = top1_eval, top5_eval, top10_eval
            torch.save(model.state_dict(), "best_model.pth")
            torch.save(
                {
                    'sample_embedding_network': model.sample_embedding_network.state_dict(),
                    'conformer': model.conformer.state_dict(),
                }, args.dataset_name + '_backbone.pth')
            
            torch.save({'attention': model.attention.state_dict(),
                        }, args.dataset_name + '_attention.pth')
            torch.save({'linear': model.linear.state_dict(),
                        }, args.dataset_name + '_linear.pth')
        print('Top 1 accuracy:  {:.4f}'.format(top1_eval))
        print('Top 5 accuracy:  {:.4f}'.format(top5_eval))
        print('Top 10 accuracy: {:.4f}'.format(top10_eval))
        print('Loss:            {:.4f}'.format(avg_loss))