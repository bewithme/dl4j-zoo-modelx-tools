package org.freeware.dl4j.modelx.tools.yolo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;


/**
* @author wenfeng xu;wechatId:italybaby;email:bewithmeallmylife@163.com
* 1. Randomly select  K centroids in all data.
* 2. Calculate the distance between each data and centroid, and classify each data into the nearest centroid.
* 3. Recalculate the centroid of the clustered data. The new centroid w of the class=the sum of all w of the class/the number of data of the class, and the new centroid h of the class =the sum of all h of the class/the number of data of the class.
* 4. Calculate the distance between each data and the new centroid , and classify each data into the new centroid nearest to the data.
* 5. Repeat 3 or 4 steps until the centroids stops changing or changes very little.
* The distance formula d = 1-IOU (single data, all centroids);
* Reference: https://github.com/experiencor/keras-yolo2/blob/master/gen_anchors.py
* Because the generating mechanism of random number in Python is different from that in java, which results in different centroid of initialization, the result of this program may be different from that calculated by gen_anchors.py, but the algorithm is the same.
* If there is no difference, the results of this program are the same as those calculated by gen_anchors.py.
 
 * 1、在所有数据中随机选择K个质心。
 * 2、计算每笔数据与质心的距离，每笔数据归类到与这笔数据距离最近的质心。
 * 3、重新计算已聚类的数据的质心,类的新质心w=该类所有w之和/该类的数据数量，类的新质心h=该类所有h之和/该类的数据数量。
 * 4、计算每笔数据与新的质心的距离，每笔数据归类到与这笔数据距离最近的新质心。
 * 5、重复3、4 步，直到质心停止变化，或是变化很小为止。
 *  距离计算公式 d=1-IOU(单笔数据,所有质心)；
 * 参考:https://github.com/experiencor/keras-yolo2/blob/master/gen_anchors.py
 * 由于python中的随机数产生机制与java的不一样，造成初始化的质心不同，所以本程序结果可能与gen_anchors.py计算的结果不同，但算法相同。
 * 如果不存在上述差异，本程序结果与gen_anchors.py计算的结果一致。
 *  
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
@Slf4j
public class YoloAnchorKmeans {
	
	private int k;
	
	private List<YoloBoundingBox> yoloBoundingBoxList;
	
	private List<YoloBoundingBoxCentroid>  boundingBoxCentroidList=new ArrayList<YoloBoundingBoxCentroid>(10);
	
	private List<YoloBoundingBox> oldYoloBoundingBoxList;
	
	private Random random=new Random();
	
	
	 public static void main(String[] args) {
			
			try {
				
				List<YoloBoundingBox> yoloBoundingBoxList = YoloBoundingBoxConverter.convertAll("/Users/wenfengxu/Downloads/dataset/people_fall_down/Annotations");
			
				Random random=new Random(123456);
				//yolo2 k=5  yolo3 k=9
				int k=9;

				YoloAnchorKmeans yoloAnchorKmeans=new YoloAnchorKmeans(yoloBoundingBoxList,random,k);
				
				yoloAnchorKmeans.init();
				    
			    double[][] priorBoxes=yoloAnchorKmeans.getPriorBoxes();
		
			    printArray(priorBoxes);
		          
				
			} catch (Exception e) {
				log.error("error:",e);
			}
			
	}


	private static void printArray(double[][] priorBoxes) {
		
		StringBuffer sb=new StringBuffer("[");
		
		for (int i = 0;i < priorBoxes.length;i++) {
		       
			   double[] item=priorBoxes[i];
			   
			   if(i!=0) {
				   sb.append(",");
			   }
			   sb.append("[");
			   
			   for(int j=0;j<item.length;j++) {
				   double n=item[j];
				   if(j!=0) {
					   sb.append(",");
				   }
				   sb.append(n);
			   }
			   sb.append("]");
			   
		   }
			sb.append("]");
			
			log.info(sb.toString());
	}

	
	public void init() {
		
		if(yoloBoundingBoxList ==null) {
			
			throw new RuntimeException("normalizeBoundingBoxList should not be null !");
		}
				
		
		this.boundingBoxCentroidList=createRandomBoundingBoxCentroidList(this.k,this.boundingBoxCentroidList);
		
	    int iterationCount=1;
	    
		while(true) {
			
			
			List<YoloBoundingBox> newYoloBoundingBoxList =cluster(this.yoloBoundingBoxList, this.boundingBoxCentroidList);
		
			
            List<YoloBoundingBoxCentroid>  newBoundingBoxCentroidList=calculateBoundingBoxCentroidList(newYoloBoundingBoxList);
			
			
			boolean retVal=compareCentroidList(this.boundingBoxCentroidList, newBoundingBoxCentroidList);
		
			
			if(retVal) {
				
				log.info("cluster end at iteration:"+iterationCount);
				
				return ;
			}
			
			this.oldYoloBoundingBoxList = newYoloBoundingBoxList;
			
			this.boundingBoxCentroidList=newBoundingBoxCentroidList;
			
			iterationCount++;
			
			
		}
		
	}

    /**
     * Randomly select K bounding boxes as centroid from all bounding boxes 
     * 在所有边界框中随机选择K个作为初始化质心
     * 
     * @param k
     * @param boundingBoxCentroidList
     * @return
     */
	private List<YoloBoundingBoxCentroid> createRandomBoundingBoxCentroidList(int k, List<YoloBoundingBoxCentroid> boundingBoxCentroidList) {
		
		if(k==0) {
			k=5;
		}
		int normalizeBoundingBoxListSize=this.yoloBoundingBoxList.size();
	
		for(int i=0;i<k;i++) {
			
			int randomNormalizeBoundingBoxIndex=random.nextInt(normalizeBoundingBoxListSize);
			
			YoloBoundingBox normalizeBoundingBox=this.yoloBoundingBoxList.get(randomNormalizeBoundingBoxIndex);
			
			YoloBoundingBoxCentroid boundingBoxCentroid=new YoloBoundingBoxCentroid();
			
			boundingBoxCentroid.setW(normalizeBoundingBox.getW());
			
			boundingBoxCentroid.setH(normalizeBoundingBox.getH());
			
			boundingBoxCentroid.setId(i);
			
			boundingBoxCentroidList.add(boundingBoxCentroid);
			
		}
		
		return boundingBoxCentroidList;
	}
	
   
	
	/**
	 * Compare centroid list
	 * 比较两组质心是否相等
	 * @param boundingBoxCentroidList
	 * @param boundingBoxCentroidListTarget
	 * @return
	 */
	private Boolean compareCentroidList(List<YoloBoundingBoxCentroid>  boundingBoxCentroidList, List<YoloBoundingBoxCentroid>  boundingBoxCentroidListTarget) {
	
		for(YoloBoundingBoxCentroid boundingBoxCentroid:boundingBoxCentroidList) {
			
			for(YoloBoundingBoxCentroid boundingBoxCentroidTarget:boundingBoxCentroidListTarget) {
				
				if(boundingBoxCentroidTarget.getId().intValue()==boundingBoxCentroid.getId().intValue()) {
					if((boundingBoxCentroidTarget.getH()!=boundingBoxCentroid.getH())||(boundingBoxCentroidTarget.getW()!=boundingBoxCentroid.getW())) {
						return false;
					}
				}
			}
		}
		
		return true;
	}
	
	
	private List<YoloBoundingBox> cluster(List<YoloBoundingBox> yoloBoundingBoxList, List<YoloBoundingBoxCentroid>  boundingBoxCentroidList) {
	    
		List<YoloBoundingBox> newYoloBoundingBoxList =new ArrayList<YoloBoundingBox>(10);
		
		for(YoloBoundingBox yoloBoundingBox : yoloBoundingBoxList) {
			
			YoloBoundingBox newYoloBoundingBox =cluster(yoloBoundingBox,boundingBoxCentroidList);
			
			newYoloBoundingBoxList.add(newYoloBoundingBox);
			
		}
		return newYoloBoundingBoxList;
	}
	
	
	
	/**
	 * 计算新质心
	 * @param normalizeBoundingBoxList
	 */
	private List<YoloBoundingBoxCentroid> calculateBoundingBoxCentroidList(List<YoloBoundingBox> normalizeBoundingBoxList) {
		
		Map<Integer,List<YoloBoundingBox>> clusterMap=new HashMap<Integer,List<YoloBoundingBox>>(10);
		
        for(YoloBoundingBox normalizeBoundingBox:normalizeBoundingBoxList) {
			
           	if(clusterMap.get(normalizeBoundingBox.getCentroidId())==null) {
        		List<YoloBoundingBox> list=new ArrayList<YoloBoundingBox>(10);
        		list.add(normalizeBoundingBox);
        		clusterMap.put(normalizeBoundingBox.getCentroidId(), list);
        	}else {
        		List<YoloBoundingBox> list=clusterMap.get(normalizeBoundingBox.getCentroidId());
           		list.add(normalizeBoundingBox);
        		clusterMap.put(normalizeBoundingBox.getCentroidId(), list);
           	}
			
		}
        
       Set<Integer> clusterIdkeys=clusterMap.keySet();
       
       List<YoloBoundingBoxCentroid>  boundingBoxCentroidList=new ArrayList<YoloBoundingBoxCentroid>(10);
       
       for(Integer clusterIdKey:clusterIdkeys) {
    	   
    	   List<YoloBoundingBox> list=clusterMap.get(clusterIdKey);
    	 
    	   double w=0.0d;
    	   
    	   double h=0.0d;
    	   
    	   for(YoloBoundingBox normalizeBoundingBox:list) {
    		   
    		   w=w+normalizeBoundingBox.getW();
    		   
    		   h=h+normalizeBoundingBox.getH();
    		   
    	   }
    	   double newCentroidW=w/list.size();
    	   
    	   double newCentroidH=h/list.size();
    	   
    	   YoloBoundingBoxCentroid boundingBoxCentroid=new YoloBoundingBoxCentroid(clusterIdKey, newCentroidW, newCentroidH);
    	   
    	   boundingBoxCentroidList.add(boundingBoxCentroid);
       }
       
       return boundingBoxCentroidList;
		
	}
	
	
	
	/**
	 * 聚类
	 * @param yoloBoundingBox
	 * @param boundingBoxCentroidList
	 */
	private YoloBoundingBox cluster(YoloBoundingBox yoloBoundingBox, List<YoloBoundingBoxCentroid>  boundingBoxCentroidList) {
		
		List<YoloBoundingBoxIou> boundingBoxIouList=calculateIOU(yoloBoundingBox, boundingBoxCentroidList);
		
		//最大的Iou即最小的距离 1-iou
		YoloBoundingBoxIou boundingBoxIou=getMaxBoundingBoxIou(boundingBoxIouList);
		
		YoloBoundingBox newYoloBoundingBox =null;
		
		try {
			 newYoloBoundingBox =(YoloBoundingBox) yoloBoundingBox.clone();
		} catch (CloneNotSupportedException e) {
			log.error("error", e);
		}
		
		//将Yolo2BoundingBox归类到与其距离最小的质心
		newYoloBoundingBox.setCentroidId(boundingBoxIou.getCentroidId());
		
		return newYoloBoundingBox;
		
	}
	
	/**
	 * 计算IOU
	 * 计算时质心中心坐标与边界框中心对齐
	 * 一共有4种情况
	 * @param yoloBoundingBox
	 * @param boundingBoxCentroidList
	 * @return
	 */
	private List<YoloBoundingBoxIou> calculateIOU(YoloBoundingBox yoloBoundingBox, List<YoloBoundingBoxCentroid> boundingBoxCentroidList){
		
		
		List<YoloBoundingBoxIou> list=new ArrayList<YoloBoundingBoxIou>(5);
		
		
		for(YoloBoundingBoxCentroid boundingBoxCentroid:boundingBoxCentroidList) {
			
			double iou=0.0f;
			
			double centroidWidth=boundingBoxCentroid.getW();
			
			double centoridHeight=boundingBoxCentroid.getH();
			
			double width= yoloBoundingBox.getW();
			
			double height= yoloBoundingBox.getH();
			
			if((centroidWidth>=width)&&(centoridHeight>=height)) {
				//边界框宽高都比质心小
				iou=(width*height)/(centroidWidth*centoridHeight);
				
			}else if((centroidWidth>=width)&&(centoridHeight<=height)) {
				
				iou=(width*centoridHeight)/(width*height+(centroidWidth-width)*centoridHeight);
				
			}else if((centroidWidth<=width)&&centoridHeight>=height) {
				
				iou=(centroidWidth*height)/(width*height+centroidWidth*(centoridHeight-height));
				
			}else {
				//质心宽高都比边界框小
				iou=(centroidWidth*centoridHeight)/(width*height);
			}
			
			YoloBoundingBoxIou boundingBoxIou=new YoloBoundingBoxIou();
			
			boundingBoxIou.setCentroidId(boundingBoxCentroid.getId());
			
			boundingBoxIou.setIou(iou);
			
			list.add(boundingBoxIou);
		}
		
    		return list;
	}
	
	/**
	 * 获取最大的Iou
	 * @param list
	 * @return
	 */
	private YoloBoundingBoxIou getMaxBoundingBoxIou(List<YoloBoundingBoxIou> list) {
		YoloBoundingBoxIou maxBoundingBoxIou=new YoloBoundingBoxIou();
		maxBoundingBoxIou.setIou(0.00f);
		for(YoloBoundingBoxIou boundingBoxIou:list) {
			if(boundingBoxIou.getIou()>maxBoundingBoxIou.getIou()) {
				maxBoundingBoxIou=boundingBoxIou;
			}
		}
		return maxBoundingBoxIou;
	}

	public YoloAnchorKmeans(int k, List<YoloBoundingBox> normalizeBoundingBoxList) {
		super();
		this.k = k;
		this.yoloBoundingBoxList = normalizeBoundingBoxList;
	}
	
	public YoloAnchorKmeans(List<YoloBoundingBox> yoloBoundingBoxList) {
		super();
		this.yoloBoundingBoxList = yoloBoundingBoxList;
	}
	
	public YoloAnchorKmeans(List<YoloBoundingBox> yoloBoundingBoxList, Random random,int k) {
		super();
		this.k=k;
		this.random=random;
		this.yoloBoundingBoxList = yoloBoundingBoxList;
	}
	
	public double[][] getPriorBoxes(){
		
		 double[][] priorBoxes=new double[this.k][2];
		
		 Collections.sort(boundingBoxCentroidList, new Comparator<YoloBoundingBoxCentroid>() {
	            @Override
	            public int compare(YoloBoundingBoxCentroid centroid, YoloBoundingBoxCentroid targetCentroid) {
	            	
	            	double a=centroid.getH()*centroid.getW();
	            	
	            	double b=targetCentroid.getH()*targetCentroid.getW();
	            	
	            	if(a==b) {
	            		
	            		return 0;
	            	}if(a>b) {
	            		
	            		return 1;
	            	}
	            	
	                return -1;
	            }
	        });
		
		for(int i=0;i<boundingBoxCentroidList.size();i++) {
			
			YoloBoundingBoxCentroid boundingBoxCentroid=boundingBoxCentroidList.get(i);
			
    		priorBoxes[i]=new double[]{boundingBoxCentroid.getW(),boundingBoxCentroid.getH()};
		}
		
		return priorBoxes;
	}
	
    
}
