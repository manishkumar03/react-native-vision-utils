package com.visionutils

import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import kotlin.math.max
import kotlin.math.min

/**
 * Label database for common ML classification datasets
 */
object LabelDatabaseAndroid {

    // MARK: - COCO 80 Labels (Detection)
    private val cocoLabels = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    )

    // COCO supercategories
    private val cocoSupercategories = listOf(
        "person", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle",
        "outdoor", "outdoor", "outdoor", "outdoor", "outdoor", "animal", "animal",
        "animal", "animal", "animal", "animal", "animal", "animal", "animal", "animal", "accessory",
        "accessory", "accessory", "accessory", "accessory", "sports", "sports", "sports", "sports",
        "sports", "sports", "sports", "sports", "sports", "sports",
        "kitchen", "kitchen", "kitchen", "kitchen", "kitchen", "kitchen", "kitchen", "food", "food",
        "food", "food", "food", "food", "food", "food", "food", "food",
        "furniture", "furniture", "furniture", "furniture", "furniture", "furniture", "electronic", "electronic",
        "electronic", "electronic", "electronic", "electronic", "appliance", "appliance", "appliance", "appliance",
        "appliance", "indoor", "indoor", "indoor", "indoor", "indoor", "indoor", "indoor"
    )

    // MARK: - COCO 91 Labels (Original with background)
    private val coco91Labels = listOf(
        "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack",
        "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    )

    // MARK: - VOC Labels (20 classes)
    private val vocLabels = listOf(
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    )

    // MARK: - CIFAR-10 Labels
    private val cifar10Labels = listOf(
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    )

    // MARK: - CIFAR-100 Labels
    private val cifar100Labels = listOf(
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
        "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
        "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
        "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
        "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
        "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
    )

    // MARK: - ImageNet Labels (abbreviated for memory efficiency)
    private val imagenetLabels = listOf(
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead", "electric ray", "stingray", "cock", "hen", "ostrich",
        "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "robin", "bulbul", "jay", "magpie", "chickadee",
        "water ouzel", "kite", "bald eagle", "vulture", "great grey owl", "European fire salamander", "common newt", "eft", "spotted salamander", "axolotl",
        "bullfrog", "tree frog", "tailed frog", "loggerhead", "leatherback turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana",
        "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard", "Gila monster", "green lizard", "African chameleon", "Komodo dragon", "African crocodile",
        "American alligator", "triceratops", "thunder snake", "ringneck snake", "hognose snake", "green snake", "king snake", "garter snake", "water snake", "vine snake",
        "night snake", "boa constrictor", "rock python", "Indian cobra", "green mamba", "sea snake", "horned viper", "diamondback", "sidewinder", "trilobite",
        "harvestman", "scorpion", "black and gold garden spider", "barn spider", "garden spider", "black widow", "tarantula", "wolf spider", "tick", "centipede",
        "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken", "peacock", "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo",
        "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "drake", "red-breasted merganser", "goose"
        // Abbreviated for memory efficiency
    )

    // MARK: - Places365 Labels (abbreviated)
    private val places365Labels = listOf(
        "airfield", "airplane_cabin", "airport_terminal", "alcove", "alley", "amphitheater", "amusement_arcade",
        "amusement_park", "apartment_building_outdoor", "aquarium", "aqueduct", "arcade", "arch", "archaelogical_excavation",
        "archive", "arena_hockey", "arena_performance", "arena_rodeo", "army_base", "art_gallery", "art_school",
        "art_studio", "artists_loft", "assembly_line", "athletic_field_outdoor", "atrium_public", "attic", "auditorium",
        "auto_factory", "auto_showroom",
        "balcony_exterior", "balcony_interior", "ball_pit", "ballroom", "bamboo_forest", "bank_vault", "banquet_hall",
        "bar", "barn", "barndoor", "baseball_field", "basement", "basketball_court_indoor", "bathroom", "bazaar_indoor",
        "bazaar_outdoor", "beach", "beach_house", "beauty_salon", "bedroom", "beer_garden", "beer_hall"
    )

    // MARK: - ADE20K Labels (150 classes for segmentation)
    private val ade20kLabels = listOf(
        "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
        "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
        "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
        "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
        "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
        "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway",
        "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench",
        "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel",
        "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television",
        "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
        "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
        "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball",
        "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
        "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
        "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"
    )

    // MARK: - Dataset descriptions
    private val descriptions = mapOf(
        "coco" to "COCO 2017 object detection labels (80 classes)",
        "coco91" to "COCO original labels with background (91 classes)",
        "imagenet" to "ImageNet ILSVRC 2012 classification labels",
        "imagenet21k" to "ImageNet-21K full classification labels",
        "voc" to "PASCAL VOC object detection labels (21 classes with background)",
        "cifar10" to "CIFAR-10 image classification labels (10 classes)",
        "cifar100" to "CIFAR-100 image classification labels (100 classes)",
        "places365" to "Places365 scene recognition labels",
        "ade20k" to "ADE20K semantic segmentation labels (150 classes)"
    )

    // MARK: - Public API

    fun getLabel(index: Int, options: ReadableMap): Any {
        val dataset = options.getString("dataset") ?: "coco"
        val includeMetadata = if (options.hasKey("includeMetadata")) options.getBoolean("includeMetadata") else false

        val labels = getLabels(dataset)

        if (index < 0 || index >= labels.size) {
            throw VisionUtilsException(
                "LABEL_ERROR",
                "Index $index out of range for dataset $dataset with ${labels.size} classes"
            )
        }

        return if (includeMetadata) {
            Arguments.createMap().apply {
                putInt("index", index)
                putString("name", labels[index])

                // Add supercategory for COCO
                if (dataset == "coco" && index < cocoSupercategories.size) {
                    putString("supercategory", cocoSupercategories[index])
                }

                // Add display name (prettified version)
                putString("displayName", labels[index].replace("_", " ").split(" ")
                    .joinToString(" ") { it.replaceFirstChar { c -> c.uppercase() } })
            }
        } else {
            labels[index]
        }
    }

    fun getTopLabels(scores: ReadableArray, options: ReadableMap): WritableArray {
        val dataset = options.getString("dataset") ?: "coco"
        val k = if (options.hasKey("k")) options.getInt("k") else 5
        val minConfidence = if (options.hasKey("minConfidence")) options.getDouble("minConfidence") else 0.0
        val includeMetadata = if (options.hasKey("includeMetadata")) options.getBoolean("includeMetadata") else false

        val labels = getLabels(dataset)

        if (scores.size() != labels.size) {
            throw VisionUtilsException(
                "LABEL_ERROR",
                "Score count (${scores.size()}) doesn't match label count (${labels.size}) for dataset $dataset"
            )
        }

        // Create indexed scores and sort by confidence
        val indexedScores = (0 until scores.size()).map { idx ->
            Pair(idx, scores.getDouble(idx))
        }.sortedByDescending { it.second }

        val results = Arguments.createArray()

        // Take top K with minimum confidence
        for (item in indexedScores.take(k)) {
            if (item.second >= minConfidence) {
                val result = Arguments.createMap().apply {
                    putInt("index", item.first)
                    putString("label", labels[item.first])
                    putDouble("confidence", item.second)

                    if (includeMetadata && dataset == "coco" && item.first < cocoSupercategories.size) {
                        putString("supercategory", cocoSupercategories[item.first])
                    }
                }
                results.pushMap(result)
            }
        }

        return results
    }

    fun getAllLabels(dataset: String): WritableArray {
        val labels = getLabels(dataset)
        val result = Arguments.createArray()
        labels.forEach { result.pushString(it) }
        return result
    }

    fun getDatasetInfo(dataset: String): WritableMap {
        val labels = getLabels(dataset)

        return Arguments.createMap().apply {
            putString("name", dataset)
            putInt("numClasses", labels.size)
            putString("description", descriptions[dataset] ?: "Unknown dataset")
            putBoolean("isAvailable", true)
        }
    }

    fun getAvailableDatasets(): WritableArray {
        val result = Arguments.createArray()
        listOf("coco", "coco91", "imagenet", "voc", "cifar10", "cifar100", "places365", "ade20k").forEach {
            result.pushString(it)
        }
        return result
    }

    // MARK: - Private Helpers

    private fun getLabels(dataset: String): List<String> {
        return when (dataset) {
            "coco" -> cocoLabels
            "coco91" -> coco91Labels
            "imagenet", "imagenet21k" -> imagenetLabels
            "voc" -> vocLabels
            "cifar10" -> cifar10Labels
            "cifar100" -> cifar100Labels
            "places365" -> places365Labels
            "ade20k" -> ade20kLabels
            else -> throw VisionUtilsException(
                "LABEL_ERROR",
                "Unknown dataset: $dataset. Available: coco, coco91, imagenet, voc, cifar10, cifar100, places365, ade20k"
            )
        }
    }
}
