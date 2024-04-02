package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
	"github.com/unixpickle/model3d/toolbox3d"
)

const (
	GridSize   = 3
	BlockSize  = 4
	InnerIters = 8

	ConnectorRadius = 0.05
	ActRadius       = 0.35
	EdgeActRadius   = 0.25

	ImageSize = 1024
	NumFrames = 96
)

var (
	ConnectorColor = render3d.NewColor(0.5)
	Color0         = model3d.XYZ(0.0, 0.5, 1.0)
	Color1         = model3d.XYZ(1.0, 0.5, 0.0)
)

type BlockActs = [BlockSize + 2][BlockSize + 2][BlockSize + 2]float64
type GridActs = [GridSize][GridSize][GridSize]BlockActs

func main() {
	os.Mkdir("frames", 0755)

	acts := AnimationGridActs()

	frameIdx := 0
	for t := 0.0; t < 2.0+1e-5; t += 2.0 / (NumFrames - 1) {
		log.Printf("Working on frame %f", t)
		spacing := math.Min(4.0, (0.5-math.Min(math.Abs(0.5-t), math.Abs(1.5-t)))*20)

		actIndex := 0
		if t <= 0.2 {
		} else if t <= 0.8 {
			frac := (t - 0.2) / 0.6
			actIndex = int(math.Round(frac*InnerIters + 1))
		} else if t < 1.0 {
			actIndex = InnerIters + 1
		} else if t < 1.2 {
			actIndex = InnerIters + 2
		} else if t <= 1.8 {
			frac := (t - 1.2) / 0.6
			actIndex = int(math.Round(frac*(InnerIters-1) + InnerIters + 2))
		} else {
			actIndex = len(acts) - 1
		}

		grid, gridColor := CreateGrid(acts[actIndex], spacing)
		gridSlice := &model3d.IntersectedSolid{
			grid,
			model3d.NewRect(
				model3d.XYZ(-100, -0.1+0.5, -100),
				model3d.XYZ(100, 0.1+0.5, 100),
			),
		}

		theta := math.Pi/4 - math.Pi/2*(1-math.Abs(1-t))
		renderer := &render3d.RayCaster{
			Camera: render3d.NewCameraAt(
				model3d.XYZ(math.Sin(theta), math.Cos(theta), 0.2).Scale(25),
				model3d.Origin,
				0.0,
			),
			Lights: []*render3d.PointLight{
				{
					Origin: model3d.XYZ(30, 30, 10.0),
					Color:  render3d.NewColor(0.8),
				},
			},
		}
		img1 := render3d.NewImage(ImageSize, ImageSize)
		img2 := render3d.NewImage(ImageSize, ImageSize)
		collider := &model3d.SolidCollider{
			Solid:               gridSlice,
			Epsilon:             0.01,
			NormalBisectEpsilon: 1e-5,
		}
		obj := &colorFuncObject{
			Object:    &render3d.ColliderObject{Collider: collider},
			ColorFunc: gridColor.RenderColor,
		}
		renderer.Render(img1, obj)
		collider.Solid = grid
		renderer.Render(img2, obj)
		img := render3d.NewImage(ImageSize*2, ImageSize)
		img.CopyFrom(img1, ImageSize, 0)
		img.CopyFrom(img2, 0, 0)

		essentials.Must(img.Save(fmt.Sprintf("frames/%03d.png", frameIdx)))
		frameIdx++
	}
}

func CreateGrid(acts GridActs, spacing float64) (model3d.Solid, toolbox3d.CoordColorFunc) {
	block := BlockSolid()
	innerBlock := model3d.IntersectedSolid{
		block,
		model3d.NewRect(model3d.Ones(-ActRadius), model3d.Ones(BlockSize-1+ActRadius)),
	}

	totalSize := block.Max().Sub(block.Min()).X*GridSize + (spacing-EdgeActRadius*2-1)*(GridSize-1)
	increment := block.Max().Sub(block.Min()).X + spacing - EdgeActRadius*2 - 1

	var solids model3d.JoinedSolid

	// We prioritize the inside of every block over the outside,
	// so that when rendering cross-sections we always see the
	// inside color and not the slow-to-update edge color when
	// two blocks are overlapping.
	var innerSolids model3d.JoinedSolid
	var innerSolidsAndColors []any
	var outerSolidsAndColors []any

	for i := 0; i < GridSize; i++ {
		z := -totalSize/2 + increment*float64(i)
		for j := 0; j < GridSize; j++ {
			y := -totalSize/2 + increment*float64(j)
			for k := 0; k < GridSize; k++ {
				x := -totalSize/2 + increment*float64(k)
				delta := model3d.XYZ(x, y, z).Sub(block.Min())
				outerSolid := model3d.TranslateSolid(block, delta)
				innerSolid := model3d.TranslateSolid(innerBlock, delta)
				solids = append(solids, outerSolid)
				colorFunc := BlockColorFunc(acts[i][j][k]).Transform(
					&model3d.Translate{Offset: delta},
				)
				innerSolids = append(innerSolids, innerSolid)
				innerSolidsAndColors = append(
					innerSolidsAndColors,
					innerSolid,
					colorFunc,
				)
				outerSolidsAndColors = append(
					outerSolidsAndColors,
					outerSolid,
					colorFunc,
				)
			}
		}
	}

	innerCf := toolbox3d.JoinedSolidCoordColorFunc(nil, innerSolidsAndColors...)
	outerCf := toolbox3d.JoinedSolidCoordColorFunc(nil, outerSolidsAndColors...)

	return solids.Optimize(), func(c model3d.Coord3D) render3d.Color {
		if innerSolids.Contains(c) {
			return innerCf(c)
		} else {
			return outerCf(c)
		}
	}
}

func BlockSolid() model3d.Solid {
	spheres := func(c model3d.Coord3D) bool {
		x := math.Round(c.X)
		y := math.Round(c.Y)
		z := math.Round(c.Z)
		if x < -1 || y < -1 || z < -1 || x > BlockSize || y > BlockSize || z > BlockSize {
			return false
		}
		dist := model3d.XYZ(x, y, z).Dist(c)
		radius := ActRadius
		if x == -1 || y == -1 || z == -1 || x == BlockSize || y == BlockSize || z == BlockSize {
			radius = EdgeActRadius
		}
		return dist < radius
	}
	var connectors model3d.JoinedSolid
	for i := 0; i < BlockSize; i++ {
		for j := 0; j < BlockSize; j++ {
			for k := 0; k < BlockSize; k++ {
				for a := -1; a <= 1; a++ {
					for b := -1; b <= 1; b++ {
						for c := -1; c <= 1; c++ {
							if a == 0 && b == 0 && c == 0 {
								continue
							}
							x, y, z := i+a, j+b, k+c
							if x < -1 || y < -1 || z < -1 || x > BlockSize || y > BlockSize || z > BlockSize {
								continue
							}
							p1 := model3d.XYZ(float64(i), float64(j), float64(k))
							p2 := model3d.XYZ(float64(x), float64(y), float64(z))
							connectors = append(connectors, &model3d.Cylinder{
								P1:     p1,
								P2:     p2,
								Radius: ConnectorRadius,
							})
						}
					}
				}
			}
		}
	}
	return model3d.JoinedSolid{
		connectors.Optimize(),
		model3d.CheckedFuncSolid(
			model3d.Ones(-(EdgeActRadius + 1)),
			model3d.Ones(BlockSize+EdgeActRadius),
			spheres,
		),
	}
}

func BlockColorFunc(acts BlockActs) toolbox3d.CoordColorFunc {
	return func(c model3d.Coord3D) render3d.Color {
		x := math.Round(c.X)
		y := math.Round(c.Y)
		z := math.Round(c.Z)
		if x < -1 || y < -1 || z < -1 || x > BlockSize || y > BlockSize || z > BlockSize {
			return render3d.NewColor(0)
		}
		dist := model3d.XYZ(x, y, z).Dist(c)
		radius := ActRadius
		if x == -1 || y == -1 || z == -1 || x == BlockSize || y == BlockSize || z == BlockSize {
			radius = EdgeActRadius
		}
		if dist > radius {
			return ConnectorColor
		}
		act := acts[int(z)+1][int(y)+1][int(x)+1]
		rgb := Color0.Scale(1 - act).Add(Color1.Scale(act))
		return render3d.NewColorRGB(rgb.X, rgb.Y, rgb.Z)
	}
}

func AnimationGridActs() []GridActs {
	grids := make([]GridActs, InnerIters*2+2)
	for i := range grids {
		rg := FullRandomActs()
		if i == len(grids)-1 {
			rg = grids[0]
		}
		if i == InnerIters+1 {
			grids[i+1] = rg
		} else if i == InnerIters+2 {
			continue
		}
		if i != 0 {
			rg = ReplaceGridInnerActs(grids[i-1], rg)
		}
		grids[i] = rg
	}
	return grids
}

func FullRandomActs() GridActs {
	side := GridSize * BlockSize
	sourceValues := make([]float64, side*side*side)
	for i := range sourceValues {
		sourceValues[i] = rand.Float64()
	}
	var result GridActs
	for i := 0; i < GridSize; i++ {
		for j := 0; j < GridSize; j++ {
			for k := 0; k < GridSize; k++ {
				var slice BlockActs
				for a := 0; a < BlockSize+2; a++ {
					for b := 0; b < BlockSize+2; b++ {
						for c := 0; c < BlockSize+2; c++ {
							globalZ := i*BlockSize + a - 1
							globalY := j*BlockSize + b - 1
							globalX := k*BlockSize + c - 1
							if globalX < 0 || globalY < 0 || globalZ < 0 ||
								globalX >= side || globalY >= side || globalZ >= side {
								slice[a][b][c] = 0.0
							} else {
								slice[a][b][c] = sourceValues[(globalZ*side+globalY)*side+globalX]
							}
						}
					}
				}
				result[i][j][k] = slice
			}
		}
	}
	return result
}

func ReplaceGridInnerActs(b, replacement GridActs) GridActs {
	for i := 0; i < GridSize; i++ {
		for j := 0; j < GridSize; j++ {
			for k := 0; k < GridSize; k++ {
				b[i][j][k] = ReplaceBlockInnerActs(b[i][j][k], replacement[i][j][k])
			}
		}
	}
	return b
}

func ReplaceBlockInnerActs(b, replacement BlockActs) BlockActs {
	for i := 1; i <= BlockSize; i++ {
		for j := 1; j <= BlockSize; j++ {
			for k := 1; k <= BlockSize; k++ {
				b[i][j][k] = replacement[i][j][k]
			}
		}
	}
	return b
}

type colorFuncObject struct {
	render3d.Object
	ColorFunc render3d.ColorFunc
}

func (c *colorFuncObject) Cast(r *model3d.Ray) (model3d.RayCollision, render3d.Material, bool) {
	rc, mat, ok := c.Object.Cast(r)
	if ok && c.ColorFunc != nil {
		p := r.Origin.Add(r.Direction.Scale(rc.Scale))
		color := c.ColorFunc(p, rc)
		mat = &render3d.LambertMaterial{
			DiffuseColor: color.Scale(0.8),
			AmbientColor: color.Scale(0.2),
		}
	}
	return rc, mat, ok
}
