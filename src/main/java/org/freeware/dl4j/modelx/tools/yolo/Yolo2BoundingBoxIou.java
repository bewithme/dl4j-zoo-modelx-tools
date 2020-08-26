package org.freeware.dl4j.modelx.tools.yolo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 
 * @author xuwenfeng
 *
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class Yolo2BoundingBoxIou implements Cloneable{
	
	private Integer centroidId;
	
	private double iou;

}
