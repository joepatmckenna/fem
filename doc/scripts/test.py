import numpy as np
import pandas as pd
import Bio
from Bio import AlignIO
import os, urllib

data_dir = '../../data/msa'

# pfam ftp faq: https://pfam.xfam.org/help#tabview=tab13
pfam_current_release = 'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release'

files = [
    'Pfam-A.full.gz',  # The full alignments of the curated families (~6GB)
    'pdbmap.gz',  # Mapping between PDB structures and Pfam domains. (~2MB)
]

for f in files:
    local = os.path.join(data_dir, f)
    if not os.path.exists(local):
        remote = os.path.join(pfam_current_release, f)
        urllib.urlretrieve(remote, local)

# protein families
pf_file = os.path.join(data_dir, 'pf.npy')
if os.path.exists(pf_file):
    pf = np.load(pf_file)
else:
    pf = []
    with open(os.path.join(data_dir, 'Pfam-A.full'), 'r') as f:
        for i, line in enumerate(f):
            if line[:7] == '#=GF AC':
                pf.append(line.split(' ')[4][:-1])
    pf = np.array(pf)
    np.save(pf_file, pf)

pdbmap_file = os.path.join(data_dir, 'pdbmap')
names = ['pdbid', 'chain', '?', 'name', 'pf', 'id', 'res']
pdbmap = pd.read_csv(
    pdbmap_file,
    sep='\t',
    engine='python',
    header=None,
    names=names,
    dtype=str)
for name in names:
    pdbmap[name] = pdbmap[name].map(lambda x: x.rstrip(';'))

pdbmap_pf = pdbmap['pf'].unique()

alignments = AlignIO.parse(os.path.join(data_dir, 'Pfam-A.full'), 'stockholm')

pf_info = []

for i, a in enumerate(alignments):

    if not pf[i].split('.')[0] in pdbmap_pf:
        continue

    pf_dir = os.path.join(data_dir, 'Pfam', pf[i])

    try:
        os.makedirs(pf_dir)
    except:
        pass

    msa = np.array(a).T
    msa = np.array([[s.lower() for s in r] for r in msa])

    ids = np.array([ai.id for ai in a])

    m = np.array([len(np.unique(s)) for s in msa])

    np.save(os.path.join(pf_dir, 'msa.npy'), msa)
    np.save(os.path.join(pf_dir, 'ids.npy'), ids)

    pf_info.append([pf[i], i, m.min(), m.max(), msa.shape[0], msa.shape[1]])
    print i, pf[i], m.min(), m.max(), msa.shape

# pf accession number
# index in 'pf' array
# min m
# max m
# residues
# sequences
np.save(os.path.join(data_dir, 'pf_info.npy'))
