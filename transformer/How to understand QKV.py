import torch
import torch.nn as nn
import torch.nn.functional as F

"""
在 PyTorch 中，**所有可学习的网络层都需要继承 `nn.Module`**。继承 `nn.Module` 可以带来以下好处：

1. **管理可学习参数**  
   由于 `BertSelfAttention` 中包含了可学习的线性层（`self.query`, `self.key`, `self.value` 和 `self.output`），这些线性层本质上都包含需要在训练过程中更新的参数权重和偏置。只有继承 `nn.Module`，PyTorch 才能自动识别和管理这些可学习参数，并在执行反向传播时为其计算梯度。

2. **与 PyTorch 训练流程集成**  
   `nn.Module` 中定义了一些关键方法（例如 `parameters()`、`children()`、`state_dict()` 等），它们可以让我们更方便地：
   - 访问或迭代网络中的所有子层（如上面提到的 `self.query`, `self.key` 等）。
   - 将参数加载或保存到文件（方便 checkpoint 的保存与恢复）。
   - 与优化器集成（通过 `model.parameters()` 就能轻松把所有子层的参数传给优化器）。

3. **构建模块化网络**  
   继承 `nn.Module` 能让你的注意力模块像其它 PyTorch 模块一样被自由地组合、调用并嵌入到更复杂的模型结构当中。比如在构建完整的 BERT 模型时，就能将这个 `BertSelfAttention` 模块作为一个子层，直接在主模型中被调用、堆叠或者重复使用。

简单来说，**继承 `nn.Module` 是为了让你的自定义层（如自注意力模块）拥有“网络模块”的身份**，从而享受 PyTorch 在管理参数、自动求梯度和模型结构组织方面提供的所有便利。
"""
class BertSelfAttention(nn.Module):
    """
    在 Python 中，**`__init__` 是类的构造方法（构造函数）**，它会在创建类实例时自动被调用，用于初始化对象的属性、执行任何需要在对象创建时就完成的操作。这个命名是 Python 语言本身所规定的「魔术方法」(Magic Method) 命名规范，而不是用户自由决定的。如果你写成别的函数名（例如 `chushijhua`），则不会在实例化时自动调用，自然也就无法充当构造函数的角色。

    此外，Python 规定类方法的第一个参数通常写作 `self`，表示对当前类实例自身的引用。你可以把 `self` 换成别的名字（比如 `this`、`obj` 等），在技术上 Python 也能识别，但这是强烈**不建议**的做法——因为 `self` 早已是社区约定俗成的习惯，用其他名称会让代码的可读性和可维护性大打折扣，也不符合 Python 提倡的「明晰、优雅」风格。

    **所以，结论就是：**
    1. `__init__` 是 Python 专门用来做构造函数的「魔术方法」，它在对象创建时自动调用，不能任意更改函数名。
    2. `self` 是对当前实例的引用，用其他名字并不会报错，但不符合约定俗成的习惯用法。
    """
    def __init__(self, config):
        """BERT自注意力机制实现
        参数:
            config - 包含以下属性的配置对象:
                hidden_size: 隐藏层维度（默认768）
                num_attention_heads: 注意力头数（默认12）
        """
        super(BertSelfAttention, self).__init__()
        """
                def __init__(self, *args, **kwargs) -> None:
                在 Python 中，`*args` 和 `**kwargs` 的作用，简单来说就是**“收集和打包不定数量的参数”**。而 `-> None` 则是 Python 3 的**函数返回类型注解**，表示这个函数（方法）不会有任何返回值（或说返回 `None`）。下面分开讲解：

        ---

        ## 1. `*args` 和 `**kwargs` 的含义

        - **`*args`**：用来接收**不定数量**的**“位置参数”**，在函数内部会被封装成一个**元组**。  
        比如，如果函数定义是 `def func(*args): ...`，然后调用时：  
        ```python
        func(10, 20, 30)
        ```  
        那么在 `func` 内部，`args` 就会是 `(10, 20, 30)`。

        - **`**kwargs`**：用来接收**不定数量**的**“关键字参数”**，在函数内部会被封装成一个**字典**。  
        比如，如果函数定义是 `def func(**kwargs): ...`，然后调用时：  
        ```python
        func(a=10, b=20, c=30)
        ```  
        那么在 `func` 内部，`kwargs` 就会是 `{"a": 10, "b": 20, "c": 30}`。

        由于你贴出来的代码是 PyTorch 内部的 `nn.Module.__init__`，它在某些情况下需要做一些**动态或兼容性处理**（可能有额外的参数，或者向后兼容老版本等）。因此会使用 `*args, **kwargs` 这种更灵活的写法，来捕获所有可能传递进来的参数，再做相应处理。

        ---

        ## 2. 为什么 `-> None`？

        这是 Python 3 中的**函数返回类型注解 (Function Annotation)**，也常被称为“类型提示 (Type Hint)”。写在函数签名的右边箭头后面，用于告知读者（以及静态类型检查器或编辑器）**函数的返回类型**。

        - `-> None` 就表示这个函数返回的类型是 `None`，或者更口语化地说“这个函数不返回任何有意义的值”。  
        - 这个类型注解并不会影响运行时行为（Python 并不会在运行时真的去强制检查返回值类型），但是**对于代码可读性、IDE 辅助、类型检查器（如 `mypy`）很有帮助**。

        比如：

        ```python
        def greet(name: str) -> None:
            print(f"Hello, {name}")
        ```

        这里就清晰地告诉我们，这个函数需要一个字符串参数 `name`，并且它不会返回任何东西（返回类型是 `None`）。

        ---

        ### 小结

        - `*args, **kwargs`：
        - `*args` 收集额外的“位置参数”，放进一个元组。
        - `**kwargs` 收集额外的“关键字参数”，放进一个字典。
        - 这样可以使函数具备更高的灵活性，能处理不确定数量或类型的参数。

        - `-> None`：
        - 这是 Python 3 的函数返回值类型注解，表明函数返回“空”或说不返回任何值。
        - 仅用于类型提示，对运行时不产生实际影响，但极大提高了可读性和可维护性。
        """
        # 初始化参数
        self.hidden_size = config.hidden_size  # 总隐藏层维度 768
        self.num_attention_heads = config.num_attention_heads  # 注意力头数 12
        self.all_head_size = self.hidden_size // self.num_attention_heads  # 每个头的维度 768/12=64

        # print(self.parameters())

        # 初始化QKV投影矩阵 (参数共享版本)
        # 每个头的维度是all_head_size，总输出维度为 num_heads * all_head_size = 768
        self.query = nn.Linear(config.hidden_size, self.all_head_size * self.num_attention_heads)
        """
                在 BERT 的原始设计中，`hidden_size = 768` 是一种**工程上的折中**和**经验选择**，并不一定非要选 768，关键是要满足几个需求：

        1. **可分成若干个注意力头**  
        BERT Base 里有 12 个注意力头，每个头的大小是 64（即 \(12 \times 64 = 768\)）。这样做保证隐藏维度能整除头数，便于多头注意力的计算和实现。

        2. **与 GPU 硬件对齐**  
        64、128、256、512、768、1024 等数字在深度学习中常被使用，往往是因为它们对 GPU 的线程/块大小比较友好，能更好地利用内存带宽和硬件并行度。虽然不是一定要用 2 的幂次方，但是使用类似 64 的倍数也能带来较好的性能。

        3. **大模型容量 vs. 计算/内存成本**  
        从参数量和计算量角度来说，768 在当时（2018 年）是一个比较适中的选择。它既能提供足够大的模型容量（足以在大语料上进行有效学习），又不会让计算量/内存需求过高。这也是 BERT Base / Large 区分的一部分设计考量（如 BERT Large 用 1024 hidden size）。

        所以 768 不是一个特殊的“魔法数字”，而是**考虑到多头注意力的分割、硬件特性以及当时的可用算力**，最终在实践中被选用的数值。并非必须要是 2 的幂，也不一定要是奇数或什么特殊质数，而是一个能**整除、对齐**且**在硬件与模型容量之间做了平衡**的数字。
        """
        self.key = nn.Linear(config.hidden_size, self.all_head_size * self.num_attention_heads)
        self.value = nn.Linear(config.hidden_size, self.all_head_size * self.num_attention_heads)

        """
                从数学维度上看，你所说的“先除后乘” 的确可以直接写成 768→768，看起来像是“多此一举”。但之所以要把 `hidden_size` 拆成 “`num_attention_heads` × `all_head_size`” 再合并回来，**主要是为了符合多头注意力的计算与代码逻辑**，并为后续步骤做铺垫。下面分几点说明：

        ---

        ## 1. 多头注意力的概念需要这样“先拆分后合并”

        在多头注意力（multi-head attention）里，我们想把一个 `hidden_size`（如 768）的向量分割成 `num_attention_heads`（如 12 个）小向量（每个大小 `all_head_size` = 64）。然后在自注意力计算时，会**把这 12 个小向量分别视作 12 个独立的头**，并行进行注意力机制计算。  

        - **“先除”**：`self.all_head_size = self.hidden_size // self.num_attention_heads`  
        - 表示每个头的维度是多少（64）。  
        - **“再乘”**：`self.all_head_size * self.num_attention_heads`  
        - 又回到 768 的总维度，只不过这一次我们已经有了“头数 × 每头大小”的概念。

        如果只写 `nn.Linear(768, 768)`，虽然最终的张量形状依然是 `[batch_size, seq_len, 768]`，但是代码里就失去了“**这一层实际上是要输出多个头**”的含义，也无法在后面轻松地“拆分 768 变成 12×64、再单独处理每个头”。换句话说，这样写更能使**模型结构、超参数（头数、头大小）**以及后续 reshape/rearrange 的流程对应起来。

        ---

        ## 2. 为后续 reshape 做铺垫

        在前向传播时，得到形如 `(batch_size, seq_len, 768)` 的输出后，一般会做一次 reshape 或 view 操作，把第三个维度 `768` 重新拆分成 `(num_attention_heads, all_head_size)` 变成 `(batch_size, seq_len, 12, 64)`。这样就可以**很自然地**对 12 个头分别做 `matmul` 等操作。

        如果你只是“名义上”写 `nn.Linear(768, 768)`，后面再手动 reshape 成 `(12, 64)` 当然也可以，但在代码可读性、灵活性（比如改成 8 头、16 头）上，都不如把它写成 `self.all_head_size * self.num_attention_heads` 这样**显示**地告诉读者：*“这里我是要 12 个头，每头 64”*。  

        ---

        ## 3. 便于模型可配置和扩展

        BERT 并不一定只有 `12` 头、`768` hidden size，你也可以遇到 `1024` hidden size、`16` 头的配置，或其他变体。如果你硬编码成 `768->768`，那就会让修改头数、hidden size 变得隐蔽、难以读懂。而采用

        ```python
        self.all_head_size = self.hidden_size // self.num_attention_heads
        nn.Linear(config.hidden_size, self.all_head_size * self.num_attention_heads)
        ```

        **自动**地适配了任意 `(hidden_size, num_attention_heads)` 的组合。只要 `hidden_size` 能整除 `num_attention_heads`（或做一些向上/向下取整策略），就能轻易套用在不同模型规模上。

        ---

        ### 小结

        - **从形状上**：确实 `all_head_size * num_attention_heads == hidden_size`，结果仍是 768。  
        - **从逻辑上**：通过显式拆分再组合，能清晰地表达“多头注意力”的结构意图，并在后续的计算步骤（reshape / permute 等）中更加自洽。  
        - **从扩展性**：保留了对任意头数、任意 hidden size 的可配置性，而不是死板地写 `nn.Linear(768, 768)`。  

        因此，看似“多此一举”，实际上是**为了让代码在概念与实现上都保持一致**，同时也方便后续多头的切分与合并。
        """

        # 输出层：将多头注意力结果转换回原始维度
        self.output = nn.Linear(self.num_attention_heads * self.all_head_size, config.hidden_size)

    """
            在 Python 中，当你在类里面定义一个普通的“实例方法”时，按照约定俗成的规则，你需要把“当前对象”作为第一个参数传进去，一般就命名为 `self`。这有以下原因：

        1. **Python 语言的设计**  
        与很多其他面向对象语言不同（比如 C++/Java/C# 有隐藏的 `this` 指针或引用），Python 并没有一个内置关键字来表示当前实例，而是通过显式的第一个参数传递给方法。  
        
        2. **约定俗成的写法**  
        Python 社区约定：在实例方法的第一个参数上使用名称 `self` 表示“当前对象”。虽然你可以换成别的名字（如 `obj`、`this`），Python 也能工作，但这是强烈**不推荐**的，因为它会给代码阅读和维护带来困惑。

        3. **访问实例属性**  
        当你在实例方法内写 `self.query`、`self.value` 这样的属性时，实际上就是在访问“当前对象”的对应属性或方法。如果没有 `self`，Python 就不知道你指的是哪一个实例。

        因此，无论是构造函数（`__init__`）还是普通方法（如 `transpose_for_scores`、`forward` 等），只要是**实例方法**，它都需要将“当前对象”作为第一个参数，而习惯上我们写成 `self`。这也是为什么你看到：

        ```python
        def transpose_for_scores(self, x):
            ...

        def forward(self, hidden_states, attention_mask=None):
            ...
        ```

        每个方法都是这样：它们都要能够访问并操作 **“self（当前实例）”** 内部的属性和状态。
    """
    def transpose_for_scores(self, x):
        """调整张量形状用于多头注意力计算
        输入形状: (batch_size, seq_len, hidden_size)
        输出形状: (batch_size, num_heads, seq_len, head_size)
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.all_head_size)
        x = x.view(*new_shape)  # 重塑形状增加头维度
        return x.permute(0, 2, 1, 3)  # 调整维度顺序为 [batch, heads, seq_len, features]

    def forward(self, hidden_states, attention_mask=None):
        """自注意力计算流程
        参数:
            hidden_states: 输入张量 (batch_size, 序列长度, 隐藏层维度)
            attention_mask: 注意力掩码 (batch_size, 序列长度)
        返回:
            attention_output: 注意力输出 (batch_size, 序列长度, 隐藏层维度)
        """

        """
                是的，当你调用例如 `nn.Linear(config.hidden_size, self.all_head_size * self.num_attention_heads)` 时，PyTorch 会在初始化这个线性层时自动创建并初始化两个参数：

        1. **权重 W**：  
        - 大小为 `[out_features, in_features]`，例如在 BERT Base 的情况就是 `[768, 768]`。  
        - 默认情况下，PyTorch 会使用一种均匀分布来初始化权重，其范围通常是 \(-\frac{1}{\sqrt{in\_features}}\) 到 \(\frac{1}{\sqrt{in\_features}}\)。

        2. **偏置 b**：  
        - 大小为 `[out_features]`，例如 `[768]`。  
        - 默认情况下，偏置通常被初始化为 0。

        这些参数是在模块创建（实例化）时自动初始化的，因此每次运行实验时，如果不设置随机种子，这些参数的初始值都会随机变化。  
        
        对于初始化方法，目前比较常见和有效的有：
        - **Xavier/Glorot 初始化**：适用于激活函数比较平滑的情况。  
        - **He/Kaiming 初始化**：特别适合带 ReLU 激活函数的网络。

        你可以使用 `torch.nn.init` 模块中的函数来手动设置初始化方法，例如：

        ```python
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        model.apply(init_weights)
        ```

        这样就能覆盖默认的初始化方式，根据你的需求选择更合适的初始化策略。
        """
        # 步骤1: 生成Q/K/V矩阵
        """
                当你将一个形状为 (2, 16, 768) 的张量传入 `nn.Linear` 层时，PyTorch 会自动把前两个维度（即批次和序列长度）当作“批量维度”，只对最后一维（768，即每个 token 的特征向量）执行线性变换。具体过程如下：

        1. **自动广播**  
        `nn.Linear` 设计时支持高维输入。它会将输入张量看作一组独立的 768 维向量，无论这些向量如何嵌套在其它维度中。

        2. **矩阵乘法**  
        对于每个 token 的 768 维向量 \(x\)，线性层计算 \(y = xW^T + b\)：
        - \(W\) 的形状为 \([out\_features, in\_features]\)（例如 \([768, 768]\)），
        - \(b\) 的形状为 \([out\_features]\)（例如 \([768]\)）。
        这样，每个 768 维向量被转换成一个新的 768 维向量。

        3. **结果形状不变**  
        因为对于每个 token 都进行了上述操作，输出张量依然保持形状 (2, 16, 768)（如果 out_features 为 768），只是每个 token 的表示经过了线性变换。

        总结：  
        - 输入是一个 2×16×768 的张量，代表 2 个句子、每个句子 16 个 token，每个 token 的表示是 768 维。  
        - 线性层会对每个 token 的 768 维向量单独做矩阵乘法操作，然后输出相同嵌套结构的张量，只不过最后一维变为你指定的输出维度。
        """
        query_layer = self.query(hidden_states)  # (batch, seq_len, hidden_size)
        """
                概念上是这样的，每个 token 的 768 维向量都会被独立地计算一次线性变换：

        - 对于单个 token，其输入是形状 (1, 768) 的向量 \( x \)。
        - 这个向量会和权重矩阵 \( W^T \)（形状为 \(768 \times 768\)）进行矩阵乘法，再加上偏置 \( b \)（形状 \(1 \times 768\)），计算 \( y = xW^T + b \) 得到一个新的 768 维向量。

        不过，在实际实现中，这个计算是**矢量化的**，即 PyTorch 会把整个 (batch_size, seq_len, 768) 的张量看作许多独立的 token 向量，利用高效的批量矩阵乘法同时处理它们。最后得到的输出仍然是一个 (batch_size, seq_len, 768) 的张量，每个 token 都对应一个计算后的向量。
        """
        key_layer = self.key(hidden_states)      # (batch, seq_len, hidden_size)
        value_layer = self.value(hidden_states)  # (batch, seq_len, hidden_size)

        # 步骤2: 重塑为多头结构
        query_layer = self.transpose_for_scores(query_layer)  # (batch, heads, seq_len, head_size)
        key_layer = self.transpose_for_scores(key_layer)      # (batch, heads, seq_len, head_size)
        value_layer = self.transpose_for_scores(value_layer)   # (batch, heads, seq_len, head_size)

        # 步骤3: 计算注意力分数 QK^T / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (batch, heads, seq_len, seq_len)
        attention_scores = attention_scores / (self.all_head_size ** 0.5)  # 缩放因子

        # 步骤4: 应用注意力掩码（用于处理padding等信息）
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(2)

        # 步骤5: Softmax归一化得到注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)  # (batch, heads, seq_len, seq_len)

        # 步骤6: 注意力权重与Value矩阵相乘
        context_layer = torch.matmul(attention_probs, value_layer)  # (batch, heads, seq_len, head_size)

        # 步骤7: 合并多头结果
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, heads, head_size)
        new_shape = context_layer.size()[:-2] + (self.num_attention_heads * self.all_head_size,)
        context_layer = context_layer.view(*new_shape)  # (batch, seq_len, hidden_size)

        # 步骤8: 最终线性投影
        attention_output = self.output(context_layer)  # (batch, seq_len, hidden_size)
        return attention_output

def test_bert_self_attention():
    """测试自注意力模块的维度正确性"""
    # 模拟BERT配置参数
    class Config:
        def __init__(self, hidden_size=768, num_attention_heads=12):
            self.hidden_size = hidden_size          # 隐藏层维度
            self.num_attention_heads = num_attention_heads  # 注意力头数

    # 初始化配置和模块
    config = Config()
    self_attention = BertSelfAttention(config)
    
    """
        在实际的自然语言处理（NLP）任务中，“**batch_size**” 和 “**序列长度**” 是非常常见的概念，下面给你一个直观的例子来说明：

    ---

    ## 1. 序列长度（seq_len）= 16

    假设我们在处理一段文本（例如一句英文）的 **token** 化结果。若这句话被分成了 16 个 token，那么这 16 个 token 对应了“序列长度 = 16”。

    例如，把句子 “*I love learning deep learning from scratch.*” 经过分词器处理成 16 个 token（这里只是示例，不是严格的 BERT 分词规则）：
    ```
    ["[CLS]", "i", "love", "learning", "deep", "learning", "from", "scratch", ".", 
    "and", "it", "is", "really", "fun", ".", "[SEP]"]
    ```
    这个长度就是 **16**。当然实际情况中，BERT 的 tokenizer 可能会产生略微不同的数量或带子词单位，但思路相同。

    ---

    ## 2. 批大小（batch_size）= 2

    “批大小”指的是**一次前向传播所处理的样本数**。如果我们同时拿了 **2** 个句子（或文本样本）拼成一个 batch，那这个 batch_size 就是 2。

    举个简单场景：
    - 第 1 条样本：“**I love learning deep learning from scratch. And it is really fun.**”
    - 第 2 条样本：“**Hello world, how are you today?**”

    如果它们经过分词、截断或填充（padding）后都变成长度为 16 的 token 序列，那么我们就可以把它们打包成一个 batch，**一起**喂给模型。此时模型的输入张量形状通常会是：

    ```
    (batch_size, seq_len) = (2, 16)
    ```

    当然，对 BERT 自注意力层来说，输入维度还包含 `hidden_size=768` 这一层，所以完整的输入形状往往是 `(2, 16, 768)`。

    ---

    ### 小结

    - **batch_size=2**：在一次前向计算里，我们同时处理 2 条句子（或样本）。
    - **seq_len=16**：这些句子在经过分词和填充后，每一句都包含 16 个 token。  
    - 当我们把这两个句子组成一个 batch 喂给模型时，数据最终会变成一个 `(2, 16, 768)` 的张量（如果输入 embedding 大小是 768），其中：
    - 2 是批大小，
    - 16 是每个句子的 token 数（序列长度），
    - 768 是每个 token 被嵌入到的向量维度（如 BERT 的 hidden size）。
    """
    # 创建测试输入 (batch_size=2, 序列长度=16, 隐藏层维度=768)
    batch_size = 2
    seq_len = 16
    # 两个句子 每个句子16个token 每个token128维度的特征向量
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    """
        在 PyTorch 中，每个继承自 `nn.Module` 的类都有一个特殊的方法 `__call__`。当你写 `output = self_attention(hidden_states)` 时，其实是调用了 `self_attention` 实例的 `__call__` 方法，而这个方法内部会自动调用你定义的 `forward` 方法，从而实现前向传播。  

    简单来说：
    - 你定义了一个 `forward` 方法（比如在 `BertSelfAttention` 类中）。
    - 当你用 `self_attention(hidden_states)` 调用实例时，实际上调用的是 `__call__` 方法。
    - `__call__` 方法负责一些额外的逻辑（比如 hook 执行），并最终调用 `forward(hidden_states)`。

    所以你不需要直接调用 `forward` 方法，而直接调用实例就能触发前向传播。
    """
    # 前向传播
    output = self_attention(hidden_states)
    
    # 验证输出维度
    print("输出形状:", output.shape)
    try:
        assert output.shape == (batch_size, seq_len, config.hidden_size), \
            f"维度错误: 期望({batch_size}, {seq_len}, {config.hidden_size}), 实际{output.shape}"
        print("BERT自注意力测试通过!")
    except AssertionError as e:
        print("测试失败:", e)

if __name__ == "__main__":
    # 运行测试用例
    test_bert_self_attention()
