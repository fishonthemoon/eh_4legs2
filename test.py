import torch
from help_4legs import get_gt_feat

def test_get_gt_feat():
    try:
        # 测试参数：sequence为"cut"，帧索引0，相机ID0
        sequence = "cut"
        frame_idx = 0
        cam_id = 0
        
        # 调用get_gt_feat获取gt特征
        gt_feat = get_gt_feat(sequence, frame_idx, cam_id)
        
        # 验证输出形状是否符合预期 (128, h, w)，这里h=512, w=640
        assert gt_feat.shape == (128, 512, 640), \
            f"特征形状错误，预期(128, 512, 640)，实际{gt_feat.shape}"
        
        # 验证数据类型是否正确
        assert isinstance(gt_feat, torch.Tensor), \
            f"输出应为torch.Tensor,实际为{type(gt_feat)}"
        
        # 验证设备是否为CPU（函数内部已指定cpu()）
        assert gt_feat.device.type == "cpu", \
            f"输出应在CPU上,实际在{gt_feat.device}"
        
        print("get_gt_feat测试通过!")
        return gt_feat
    
    except FileNotFoundError as e:
        print(f"测试失败：{e}")
    except Exception as e:
        print(f"测试过程中发生错误：{e}")

if __name__ == "__main__":
    # 执行测试并获取特征（如果成功）
    test_feat = test_get_gt_feat()