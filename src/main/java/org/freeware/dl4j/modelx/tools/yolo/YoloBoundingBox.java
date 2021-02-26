package org.freeware.dl4j.modelx.tools.yolo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author xuwenfeng
 * Yolo2边界框
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class YoloBoundingBox implements Cloneable{
	
	private String fileName;
	
	private String label;
	
	private int labelIndex=-1;
	
	private Integer centroidId;
	
	private double x;
	
	private double y;
	
	private double w;
	
	private double h;
	
	private VocBoundingBox boundingBox;
	
	@Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();
    }
}
