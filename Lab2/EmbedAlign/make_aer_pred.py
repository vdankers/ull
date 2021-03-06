import torch
from torch.autograd import Variable

def make_pred(english_data, french_data, pred_file, w2i_e, w2i_f, decoder,
              encoder):
    """Create NAACL format predictions for alignments.

    Args:
        english_data: file with English sentences
        french_data: file with French sentences
        pred_file: file to save predictions to
        w2i_e: dictionary mapping English words to indices
        w2i_f: dictionary mapping French words to indices
        decoder: Embed Align decoder part (generative model)
        encoder: Embed Align encoder part (inference model)

    """
    # Read data
    english_lines = []
    french_lines = []
    with open(english_data, 'r') as fe, open(french_data, 'r') as ff:
        for line in fe:
            english_lines.append(line.split())
        for line in ff:
            french_lines.append(line.split())

    # Prepare tensors / variables to input to the model
    english_lines = [Variable(torch.LongTensor([[w2i_e[w] for w in x]]).cuda())
                     for x in english_lines]
    french_lines = [Variable(torch.LongTensor([[w2i_f[w] for w in x]]).cuda())
                    for x in french_lines]

    # Write alignments to file
    with open(pred_file, 'w') as fileee:
        for i, (e,f) in enumerate(zip(english_lines, french_lines)):
            mu, sigma = encoder.forward(e)
            alignments = decoder.forward(mu, sigma, e, f, True)
            for j, a in enumerate(alignments):
                a=int(a) # just to be safe
                assert 0 < a+1 <= len(e[0]), '{} not in range 0 and {}'.format(
                    a, len(e[0])
                )
                fileee.write('{} {} {}\n'.format(i+1, int(a)+1, j+1))


if __name__ == '__main__':
    import pickle
    import aer
    
    print('loading model')
    with open('corp_enc_dec.pickle', r) as f:
        corpus, encoder, decoder = pickle.load(f)
    print('making predictions')
    make_pred('./../data/wa/test.en', './../data/wa/test.fr',
              './alignment.pred', corpus, decoder, encoder)
    print('score:')
    aer.test('./../data/wa/test.naacl', './alignment.pred')