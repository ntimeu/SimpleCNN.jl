module SimpleCNN
	import Flux

	module Utils
		import Flux
		import MLDatasets: FashionMNIST
		import ProgressMeter: BarGlyphs, next!, Progress

		function train(model, loader::Flux.Data.DataLoader, loss_fn, optimizer)
			num_images = size(loader.data[1], 4)
			num_batches = ceil(UInt32, num_images / loader.batchsize)

			Flux.trainmode!(model)
			parameters = Flux.params(model)

			bar = Progress(
				num_batches,
				dt=1.0;
				barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
				barlen=20,
				showspeed=true
			)

			for (X, y) in loader

				grads = Flux.gradient(parameters) do
					loss_fn(model(X), y)
				end

				Flux.Optimise.update!(optimizer, parameters, grads)
				next!(bar)
			end
		end

		function test(model, loader::Flux.Data.DataLoader, loss_fn)
			num_images::UInt64 = size(loader.data[1], 4)
			num_batches::UInt64 = ceil(UInt64, num_images / loader.batchsize)

			loss::Float32 = 0.0
			corrects::UInt64 = 0

			Flux.testmode!(model)

			for (X, y) in loader
				ŷ = model(X)

				loss += loss_fn(ŷ, y)
				corrects += reduce(+, Flux.onecold(ŷ) .== Flux.onecold(y))
			end

			println("loss: ", loss / num_batches,
					", accuracy: ", corrects, "/", num_images)
		end

		function get_training_data(
				bs::Int64;
				shuffle::Bool=true,
				partial::Bool=true)::Flux.DataLoader
			x = Flux.unsqueeze(FashionMNIST.traintensor(Float32, 1:60000), 3)
			y = Flux.onehotbatch(FashionMNIST.trainlabels(1:60000), 0:9)

			return Flux.DataLoader(
				(x, y);
				batchsize=bs,
				shuffle=shuffle,
				partial=partial
			)
		end

		function get_testing_data(
				bs::Int64;
				shuffle::Bool=false,
				partial::Bool=true)::Flux.DataLoader
			x = Flux.unsqueeze(FashionMNIST.testtensor(Float32, 1:10000), 3)
			y = Flux.onehotbatch(FashionMNIST.testlabels(1:10000), 0:9)

			return Flux.DataLoader(
				(x, y);
				batchsize=bs,
				shuffle=shuffle,
				partial=partial
			)
		end
	end

	function main()
		model = Flux.Chain(
			Flux.DepthwiseConv((3, 3), 1 => 64, Flux.leakyrelu; pad=1),
			Flux.MaxPool((2, 2)),
			Flux.flatten,
			Flux.Dense(14^2 * 64, 512, Flux.leakyrelu),
			Flux.Dropout(5e-1),
			Flux.Dense(512, 512, Flux.leakyrelu),
			Flux.Dropout(5e-1),
			Flux.Dense(512, 10)
		)

		loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
		opt = Flux.Optimise.ADAM(1e-3)

		train_loader::Flux.DataLoader = Utils.get_training_data(64)
		test_loader::Flux.DataLoader = Utils.get_testing_data(64)

		for epoch::UInt64 in 1:10
			Utils.train(model, train_loader, loss, opt)
			Utils.test(model, test_loader, loss)
		end
	end
end
