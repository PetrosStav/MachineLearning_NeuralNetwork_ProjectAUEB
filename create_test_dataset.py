with open('test_dataset.txt', 'w') as outfile:
    for i in range(0, 100):
        outfile.write('%d--%d\n' % (i, i % 2))