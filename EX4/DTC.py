import pandas as pd
import numpy as np
import random

class Node:
    def __init__(self, value, leaf=False, split=None):
        self._value = value
        self._edges = []
        self._leaf=leaf
        self._split=split

    def __repr__(self):
        if len(self._edges):
            return f'{self._value} --> {self._edges}'
        else:
            return f'{self._value}'

    @property
    def value(self):
        return self._value

    @property
    def split(self):
        return self._split

    @property
    def leaf(self):
        return self._leaf

    def add_edge(self, edge):
        self._edges.append(edge)

    def find_edge(self, value):
        return next(edge for edge in self._edges if edge.value == value)

class Edge:
    def __init__(self, value):
        self._value = value
        self._node = None

    def __repr__(self):
        return f'{self._value} --> {self._node}'

    @property
    def value(self):
        return self._value

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, node):
        self._node = node
            

class DecisionTreeClassifier():
    def __init__(self, train_set, test_set):
        self.train_set=train_set
        self.test_set=test_set

    def DecisionTreeLearning(self, examples, attrbutes, parent_examples):
        if examples.empty:
            return self.PLURALITY_VALUE(parent_examples)
        elif max(examples['Survived'].value_counts()) == examples['Survived'].count():
            if examples['Survived'].iloc[0]==0:
                return Node(0, leaf=True)
            else:
                return Node(1, leaf=True)
        elif not attrbutes:
            return self.PLURALITY_VALUE(examples)
        else:
            max_gain=0
            max_gain_att=attrbutes[0]
            best_split=None
            for att in attrbutes:
                gain, split = self.IMPORTANCE(att, examples)
                if gain >= max_gain:
                    max_gain=gain
                    max_gain_att=att
                    best_split=split
                elif split is not None:
                    split=None
            print('max_gain_att: ', max_gain_att)
            attrbutes.remove(max_gain_att)
            if best_split is None:
                node=Node(max_gain_att)
                values=self.train_set[max_gain_att].unique()
                for val in values:
                    exs=examples[examples[max_gain_att]==val]
                    edge=Edge(val)
                    node.add_edge(edge)
                    edge.node=self.DecisionTreeLearning(exs, attrbutes, examples)
            else:
                node=Node(max_gain_att, split=best_split)
                for part in best_split:
                    exs=examples.copy()
                    #print('Size0', exs.size)
                    for val in part:
                        exs=exs[exs[max_gain_att]==val]
                    #print('Size1', exs.size)
                    edge=Edge(part)
                    node.add_edge(edge)
                    edge.node=self.DecisionTreeLearning(exs, attrbutes, examples)
        return node

    def DecisionTreePrediction(self,tree):
        true_outcome=list(self.test_set['Survived'])
        predicted_outcome=[]
        #values=self.train_set[att].unique()
        for i, row in self.test_set.iterrows():
            att=tree.value
            check_tree=tree
            check=True
            while check:
                split=check_tree.split
                val=row[att]
                c=row['Survived']
                if split is not None:
                    vals=[]
                    for s in split:
                        if val in s:
                            vals.append(s)
                    val=vals[random.choice(list(enumerate(vals)))[0]]
                if  check_tree.find_edge(val).node.leaf:
                    predicted_outcome.append(check_tree.find_edge(val).node.value)
                    check=False
                else:
                    check_tree=check_tree.find_edge(val).node
                    att=check_tree.value
        return sum(1 for x,y in zip(true_outcome,predicted_outcome) if x == y) / len(true_outcome)   
            
    def IMPORTANCE(self, a, examples):
        all_values=list(self.train_set[a].unique())
        for val in list(self.test_set[a].unique()):
            if val not in all_values:
                all_values.append(val)
        all_values.sort()
        # Find total positive and negative samples 
        p=examples['Survived'][examples['Survived']==1].value_counts().iloc[0]
        n=examples['Survived'][examples['Survived']==0].value_counts().iloc[0]
        # Find positive and negative for attributes 
        count=examples.groupby(['Survived', a]).size()
        count=count.to_frame(name = 'size').reset_index()
        values=count[a].unique()
        rem=0
        gain=-100
        split=None
        
        # Categorical attributes
        if len(all_values) < 4:
            pk=0
            nk=0
            for val in all_values:
                pk_check=count[(count[a] == val) & (count['Survived'] == 1)]
                nk_check=count[(count[a] == val) & (count['Survived'] == 0)]
                if pk_check.size != 0: pk=pk_check.iloc[0]['size']
                if nk_check.size != 0: nk = nk_check.iloc[0]['size']
                rem += ((pk+nk)/(n+p))*self.B(pk/(pk+nk))
            gain = self.B(p/(p+n)) - rem
        
        # Constinous attributes 
        else:
            partitions=[]
            for i in range(1, len(all_values)-1, 1):
                for partition in self.get_partitions(all_values, num_partitions=i):
                    partitions.append(partition)

            for partition in partitions:
                split_gain=-100
                rem=0
                for values in partition:
                    pk=0
                    nk=0
                    for val in values:
                        pk_check=count[(count[a] == val) & (count['Survived'] == 1)]
                        nk_check=count[(count[a] == val) & (count['Survived'] == 0)]
                        if pk_check.size != 0: pk+=pk_check.iloc[0]['size']
                        if nk_check.size != 0: nk+=nk_check.iloc[0]['size']
                    if (pk==0 and nk==0):
                        select=bool(random.getrandbits(1))
                        if select:
                            pk=1
                        else:
                            nk=1

                    rem += ((pk+nk)/(n+p))*self.B(pk/(pk+nk))
                split_gain = self.B(p/(p+n)) - rem
                if split_gain > gain: 
                    gain=split_gain
                    split=partition
        return gain, split 

    
    def get_partitions(self, values, num_partitions):
        num_values = len(values)
        groups = []

        def generate_partitions(i):
            if i >= num_values:
                yield list(map(tuple, groups))
            else:
                if num_values - i > num_partitions - len(groups):
                    for group in groups:
                        group.append(values[i])
                        yield from generate_partitions(i + 1)
                        group.pop()

                if len(groups) < num_partitions:
                    groups.append([values[i]])
                    yield from generate_partitions(i + 1)
                    groups.pop()        
        return generate_partitions(0)


    def PLURALITY_VALUE(self, examples):
        val0=examples['Survived'].value_counts()[0]
        val1=examples['Survived'].value_counts()[1]
        
        if val0 > val1:
            node=Node(0, leaf=True)
        elif val0 < val1:
            node=Node(1, leaf=True)
        else:
            select=bool(random.getrandbits(1))
            if select: 
                node=Node(1, leaf=True)
            else:
                node=Node(0, leaf=True)
        
        return node
       
    def B(self, q):
        if q==1:
            return 1
        elif q==0:
            return 0
        else:
            return -(q*np.log2(q)+(1-q)*np.log2(1-q))


if __name__ == "__main__":
    # Read data 
    train_set=pd.read_csv('train.csv')
    test_set=pd.read_csv('test.csv')
    parent_examples=pd.DataFrame()

    attributes = list(train_set.columns.values)
    attributes.remove('Survived')
    attributes.remove('Name')
    attributes.remove('Cabin')
    attributes.remove('Fare')
    attributes.remove('Ticket')
    attributes.remove('Age')
    #attributes.remove('SibSp')
    #attributes.remove('Parch')
    dtc = DecisionTreeClassifier(train_set, test_set)
    tree = dtc.DecisionTreeLearning(train_set, attributes, parent_examples)
    print(tree)
    accuracy = dtc.DecisionTreePrediction(tree)
    print('Accuracy: ', accuracy)
    





