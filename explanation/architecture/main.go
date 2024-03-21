package main

import (
	"math"
	"math/rand"

	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
	"github.com/unixpickle/model3d/toolbox3d"
)

const (
	GridSize  = 4
	BlockSize = 4

	ConnectorRadius = 0.05
	ActRadius       = 0.35
	EdgeActRadius   = 0.25
)

var ConnectorColor = render3d.NewColor(0.5)

type BlockActs = [BlockSize + 2][BlockSize + 2][BlockSize + 2]float64

func main() {
	grid := GridSolid()
	gridColor := GridColorFunc(RandomActs())
	// mesh, interior := model3d.DualContourInterior(grid, 0.1, false, false)
	collider := &model3d.SolidCollider{
		Solid:               grid,
		Epsilon:             0.01,
		NormalBisectEpsilon: 1e-5,
	}
	// colorFn := toolbox3d.JoinedSolidCoordColorFunc(interior, grid, gridColor)
	// render3d.SaveRandomGrid("rendering.png", mesh, 3, 3, 300, colorFn.RenderColor)
	render3d.SaveRandomGrid("rendering.png", collider, 3, 3, 300, gridColor.RenderColor)
}

func GridSolid() model3d.Solid {
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
			model3d.XYZ(-2, -2, -2),
			model3d.XYZ(BlockSize+2, BlockSize+2, BlockSize+2),
			spheres,
		),
	}
}

func GridColorFunc(acts BlockActs) toolbox3d.CoordColorFunc {
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
		act := acts[int(x)+1][int(y)+1][int(z)+1]
		c1 := model3d.XYZ(1.0, 0.2, 0.3)
		c2 := model3d.XYZ(0.5, 0.5, 1.0)
		return c1.Scale(act).Add(c2.Scale(1 - act))
	}
}

func RandomActs() BlockActs {
	var res BlockActs
	for i := 0; i < BlockSize+2; i++ {
		for j := 0; j < BlockSize+2; j++ {
			for k := 0; k < BlockSize+2; k++ {
				res[i][j][k] = rand.Float64()
			}
		}
	}
	return res
}
