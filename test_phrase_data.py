from gensim.models import KeyedVectors

w2v = KeyedVectors.load(os.getcwd().replace('C:\Users\86186\Desktop\研一\论文\知识图谱\dataset\数据')
w2v.init_sims(replace=True)
print(self.w2v[onto_obj])
