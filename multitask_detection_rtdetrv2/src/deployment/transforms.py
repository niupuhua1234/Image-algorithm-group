# 部署加速变换：融合推理分支与归一化层，并提前筛选进入框回归头的查询。
# 这些变换用于部署副本；训练时仍使用原始网络和原有损失。
"""Deployment transforms extracted unchanged from the verified benchmark."""
import copy
import types
import torch

# 先按全图候选分数选 top-k，再仅对入选查询计算框回归；编码器仍处理全图特征。
def topk_input(self,memory,spatial_shapes,denoising_class=None,denoising_bbox_unact=None):
    # 查询筛选变换仅支持推理，训练中的去噪查询和梯度路径继续使用原实现。
    if self.training or denoising_class is not None or denoising_bbox_unact is not None:
        raise RuntimeError('Inference-only top-k optimization')
    assert getattr(self,'query_select_method','default')=='default'
    bs=memory.shape[0]
    anchors,valid=self.anchors.to(memory.device),self.valid_mask.to(memory.device)
    output=self.enc_output(valid.to(memory.dtype)*memory)
    # 全图候选仍计算分类分数，依据同一 top-k 规则筛出要进行框回归的查询。
    logits=self.enc_score_head(output)
    indices=torch.topk(logits.max(-1).values,self.num_queries,dim=1).indices
    selected=output.gather(1,indices.unsqueeze(-1).expand(-1,-1,output.shape[-1]))
    anchor=anchors.expand(bs,-1,-1).gather(1,indices.unsqueeze(-1).expand(-1,-1,4))
    # 昂贵的框回归头只处理入选的 num_queries 个特征，减少部署计算量。
    refs=self.enc_bbox_head(selected)+anchor
    learned=getattr(self,'learnt_init_query',getattr(self,'learn_query_content',False))
    content=self.tgt_embed.weight.unsqueeze(0).tile([bs,1,1]) if learned else selected.detach()
    if hasattr(self,'query_select_method'):
        return content,refs.detach(),[],[]
    return content,refs.detach(),refs.sigmoid(),logits.gather(1,indices.unsqueeze(-1).expand(-1,-1,logits.shape[-1]))


# 利用推理时固定的均值和方差，将 BN/FrozenBN 的缩放和平移折叠进前一层卷积。
def fuse_norms(model):
    def is_norm(m):
        return isinstance(m,torch.nn.BatchNorm2d) or type(m).__name__=='FrozenBatchNorm2d'
    # 折叠公式：新权重 = 原权重×缩放，新偏置 = (原偏置−均值)×缩放+归一化偏置。
    def fold(conv,norm):
        out=copy.deepcopy(conv)
        scale=norm.weight.detach()*torch.rsqrt(norm.running_var+norm.eps)
        bias=conv.bias.detach() if conv.bias is not None else torch.zeros_like(norm.running_mean)
        out.weight=torch.nn.Parameter(conv.weight.detach()*scale[:,None,None,None],requires_grad=False)
        out.bias=torch.nn.Parameter((bias-norm.running_mean)*scale+norm.bias,requires_grad=False)
        return out
    for m in list(model.modules()):
        for name in ('norm','bn'):
            conv,norm=getattr(m,'conv',None),getattr(m,name,None)
            if isinstance(conv,torch.nn.Conv2d) and is_norm(norm):
                m.conv=fold(conv,norm); setattr(m,name,torch.nn.Identity())
        if isinstance(m,torch.nn.Sequential):
            keys=list(m._modules)
            for a,b in zip(keys,keys[1:]):
                if isinstance(m._modules[a],torch.nn.Conv2d) and is_norm(m._modules[b]):
                    m._modules[a]=fold(m._modules[a],m._modules[b]);m._modules[b]=torch.nn.Identity()


# 使用矩阵乘法、掩码和 softmax 表达注意力，供旧版 PyTorch 的 ONNX 导出使用。
def attention(query,key,value,attn_mask=None,dropout_p=0.,is_causal=False):
    scores=torch.matmul(query,key.transpose(-2,-1))*(query.size(-1)**-.5)
    if is_causal:
        mask=torch.ones(scores.size(-2),scores.size(-1),dtype=torch.bool,device=scores.device).tril()
        scores=scores.masked_fill(~mask,float('-inf'))
    if attn_mask is not None:
        scores=scores.masked_fill(~attn_mask,float('-inf')) if attn_mask.dtype==torch.bool else scores+attn_mask
    # 导出路径使用 eval 模式，注意力 dropout 不启用。
    return torch.matmul(torch.softmax(scores,dim=-1),value)


# 先融合可重参数化分支，再融合归一化层，最后绑定部署专用的查询筛选方法。
def optimize(model):
    model.deploy()
    fuse_norms(model)
    model.decoder._get_decoder_input=types.MethodType(topk_input,model.decoder)
    return model.eval()
