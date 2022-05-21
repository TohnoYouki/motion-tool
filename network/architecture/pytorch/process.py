import torch
from .recoder import AverageMeter

def evaluate_process(task, dataloader):
    task.eval()
    batch, size = dataloader.batch_size, len(dataloader.dataset)
    recorder, avg_metric = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            reports, metric = task.evaluate(batch_data)
            batch_size = min(size - i * batch, batch)  
            recorder.update(reports, batch_size)
            avg_metric.update([metric], batch_size)
    for i in range(len(recorder.avg)):
        print('Evaluate_Loss {:d}, Loss: {:.6f}'.format(i, recorder.avg[i]))
    avg_metric = avg_metric.avg[0]
    print('Evaluate Metric: {:.6f}'.format(avg_metric))
    return avg_metric

def test_process(task, dataloader):
    task.eval()
    batch, size = dataloader.batch_size, len(dataloader.dataset)
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            index = range(i * batch, min((i + 1) * batch, size))
            task.test(batch_data, index)

def train_epoch(pipeline, recoder, traindata, valdata):
    print('Epoch{:d}'.format(recoder.epoch))
    pipeline.task.train()
    recorder, avg_loss = AverageMeter(), AverageMeter()
    batch, size = traindata.batch_size, len(traindata.dataset)
    for i, batch_data in enumerate(traindata):
        reports, loss = pipeline.step(batch_data)
        recorder.update(reports, min(size - i * batch, batch))
        avg_loss.update([loss], min(size - i * batch, batch))
    pipeline.update()
    for i in range(len(recorder.avg)):
        print('Train_Loss {:d}, Loss: {:.6f}'.format(i, recorder.avg[i]))
    train_loss = avg_loss.avg[0]
    avg_loss = avg_loss.avg[0]
    print('Train Tol Loss: {:.6f}'.format(avg_loss))
    val_loss = evaluate_process(pipeline.task, valdata)
    recoder.step(train_loss, val_loss)
    state = pipeline.state_dict()
    state = recoder.state_dict(state)
    return recoder.exit_train(), recoder.if_best, state