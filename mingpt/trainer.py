"""Trainer code, simplified from Andrej Karpathy's minGPT"""
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64

    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, optimizer, train_dataset, config, test_dataset=None):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.config = config
        self.test_dataset = test_dataset
        self.train_loss = None

    def save_checkpoint(self, epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        
        # save everything we need
        save_str = 'model_{}_{}'.format(epoch, self.config.ckpt_path)
        print('Saving to:', save_str)
        torch.save({'model_state_dict': raw_model.state_dict(), 
                    'train_loss': self.train_loss,
                    'clusters': self.train_dataset.clusters,
                    'model_config': raw_model.model_config,
                    }, save_str)

    def train(self, args):
        model, optimizer, config = self.model, self.optimizer, self.config

        sampler = DistributedSampler(self.train_dataset, seed=args.seed) if args.distributed else None
        loader = DataLoader(self.train_dataset, shuffle=(not args.distributed), pin_memory=True, sampler=sampler, batch_size=config.batch_size, num_workers=config.num_workers)

        # only needed for random noise experiments
        if self.test_dataset is not None:
            test_loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True, batch_size=30, num_workers=config.num_workers)  # adjust bs as necessary
            train_loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True, batch_size=30, num_workers=config.num_workers)  # adjust bs as necessary
            accuracies = []

        for epoch in range(args.start_epoch, config.max_epochs):
            
            model.train()
            
            if args.distributed: 
                loader.sampler.set_epoch(epoch)

            losses = []

            for _, (x, y) in enumerate(loader):
                # place data on the correct device
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

                # forward the model
                _, loss, _ = model(x, y)  # first output returns logits, last one returns unreduced losses
                losses.append(loss.item())

                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                optimizer.step()

            self.train_loss = float(np.mean(losses))
            print('Epoch:', epoch, '|', 'Training loss:', self.train_loss)

            # save trained model, clusters, and final train loss
            if args.distributed:
                if args.rank == 0:
                    self.save_checkpoint(epoch)
            else:
                self.save_checkpoint(epoch)

            # only needed for random noise experiments
            if self.test_dataset is not None:
                model.eval()

                with torch.no_grad():
                    unreduced_train_losses = []
                    unreduced_test_losses = []

                    # losses on train data
                    for it, (x, y) in enumerate(train_loader):
                        if it==10:
                            break

                        # place data on the correct device
                        x = x.cuda(non_blocking=True)
                        y = y.cuda(non_blocking=True)

                        # forward the model
                        _, _, unreduced_loss = model(x, y)  # the first output returns the logits, the second mean loss over batch
                        unreduced_train_losses.append(unreduced_loss.cpu().numpy())
                    unreduced_train_losses = np.concatenate(unreduced_train_losses)

                    # losses on test data
                    for it, (x, y) in enumerate(test_loader):
                        if it==10:
                            break

                        # place data on the correct device
                        x = x.cuda(non_blocking=True)
                        y = y.cuda(non_blocking=True)

                        # forward the model
                        _, _, unreduced_loss = model(x, y)
                        unreduced_test_losses.append(unreduced_loss.cpu().numpy())
                    unreduced_test_losses = np.concatenate(unreduced_test_losses)

                    accuracy = np.mean(unreduced_train_losses < unreduced_test_losses)
                    print('Epoch:', epoch, 'Accuracy:', accuracy)
                    accuracies.append(accuracy)
                    
                    np.save('acc_{}_{}.npy'.format(args.pretrain_data, args.seed), np.array(accuracies))