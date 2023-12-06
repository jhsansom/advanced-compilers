from train_embed2 import RealGraphDataset, GraphDataset
    
real_data = RealGraphDataset()
fake_data = GraphDataset(1000)

avg_feature = 0
for elem in real_data:
    avg_feature += elem.calculate_feature()
avg_feature /= len(real_data)
print(f'Real data has feature = {avg_feature}')

avg_feature = 0
for i in range(len(fake_data)):
    elem = fake_data[i]
    avg_feature += elem.calculate_feature()
avg_feature /= len(fake_data)
print(f'Fake data has feature = {avg_feature}')