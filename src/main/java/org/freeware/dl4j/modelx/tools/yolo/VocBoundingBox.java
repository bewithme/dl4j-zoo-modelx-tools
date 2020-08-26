package org.freeware.dl4j.modelx.tools.yolo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class VocBoundingBox {
	
	private String label;
	private int xmin;
	private int xmax;
	private int ymin;
	private int ymax;
	private ImageSize imageSize;
	

}
