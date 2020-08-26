package org.freeware.dl4j.modelx.tools.yolo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 质心
 * @author xuwenfeng
 *
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class Yolo2BoundingBoxCentroid {
	
	
	private Integer id;
	private double w;
	private double h;

}
