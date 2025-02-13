from utils.util import load_dataset, save_dataset
import os

def convert_pkl_structure(orig_path, new_path, instance_id=0):
    """
    参数说明：
    orig_path: 原始文件路径（如：/output/batch0-sample0/route_info.pkl）
    new_path: 转换后保存路径
    instance_id: 要分配的实例ID（默认0）
    """
    # 加载原始数据
    raw_data = load_dataset(orig_path)
    
    # 提取关键字段
    converted_data = {
        instance_id: {
            'custm_coords': raw_data['custm_coords'],
            'depot_coords': raw_data['depot_coords'],
            'route': {
                # 按车辆ID组织路径
                veh_id: route 
                for veh_id, route in enumerate(raw_data['route'])
            }
        }
    }
    
    # 自动创建目录
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    
    # 保存新格式
    save_dataset(converted_data, new_path)
    print(f"转换完成 → {new_path}")

# 使用示例
convert_pkl_structure(
    orig_path="/Users/dxk/code/2/evrp-eps/output/batch0-sample0/route_info.pkl",
    new_path="/Users/dxk/code/2/evrp-eps/output/batch0-sample0/converted_routes.pkl",
    instance_id=0  # 对应--instance参数的值
)