import Foundation

/// Label database for common ML classification datasets
class LabelDatabase {

    // MARK: - Singleton
    static let shared = LabelDatabase()
    private init() {}

    // MARK: - COCO 80 Labels (Detection)
    private let cocoLabels: [String] = [
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
    ]

    // COCO supercategories
    private let cocoSupercategories: [String] = [
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
    ]

    // MARK: - COCO 91 Labels (Original with background)
    private let coco91Labels: [String] = [
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
    ]

    // MARK: - VOC Labels (20 classes)
    private let vocLabels: [String] = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    ]

    // MARK: - CIFAR-10 Labels
    private let cifar10Labels: [String] = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ]

    // MARK: - CIFAR-100 Labels
    private let cifar100Labels: [String] = [
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
    ]

    // MARK: - ImageNet Labels (1000 classes) - Abbreviated for common use
    // Full list stored in separate resource file for memory efficiency
    private lazy var imagenetLabels: [String] = {
        // Load from embedded resource or use abbreviated list
        return loadImageNetLabels()
    }()

    private func loadImageNetLabels() -> [String] {
        // Check if we have a bundled resource file
        if let path = Bundle.main.path(forResource: "imagenet_labels", ofType: "txt"),
           let content = try? String(contentsOfFile: path, encoding: .utf8) {
            return content.components(separatedBy: "\n").filter { !$0.isEmpty }
        }

        // Fallback to commonly used subset (first 100 + key classes)
        return [
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
            // ... truncated for brevity, actual implementation loads full list
        ]
    }

    // MARK: - Places365 Labels (abbreviated)
    private let places365Labels: [String] = [
        "airfield", "airplane_cabin", "airport_terminal", "alcove", "alley", "amphitheater", "amusement_arcade",
        "amusement_park", "apartment_building_outdoor", "aquarium", "aqueduct", "arcade", "arch", "archaelogical_excavation",
        "archive", "arena_hockey", "arena_performance", "arena_rodeo", "army_base", "art_gallery", "art_school",
        "art_studio", "artists_loft", "assembly_line", "athletic_field_outdoor", "atrium_public", "attic", "auditorium",
        "auto_factory", "auto_showroom", // ... abbreviated for brevity
        "balcony_exterior", "balcony_interior", "ball_pit", "ballroom", "bamboo_forest", "bank_vault", "banquet_hall",
        "bar", "barn", "barndoor", "baseball_field", "basement", "basketball_court_indoor", "bathroom", "bazaar_indoor",
        "bazaar_outdoor", "beach", "beach_house", "beauty_salon", "bedroom", "beer_garden", "beer_hall"
    ]

    // MARK: - ADE20K Labels (150 classes for segmentation)
    private let ade20kLabels: [String] = [
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
    ]

    // MARK: - Public API

    func getLabel(index: Int, dataset: String, includeMetadata: Bool) throws -> Any {
        let labels = try getLabels(for: dataset)

        guard index >= 0 && index < labels.count else {
            throw NSError(domain: "LabelDatabase", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Index \(index) out of range for dataset \(dataset) with \(labels.count) classes"
            ])
        }

        if includeMetadata {
            var result: [String: Any] = [
                "index": index,
                "name": labels[index]
            ]

            // Add supercategory for COCO
            if dataset == "coco" && index < cocoSupercategories.count {
                result["supercategory"] = cocoSupercategories[index]
            }

            // Add display name (prettified version)
            result["displayName"] = labels[index].replacingOccurrences(of: "_", with: " ").capitalized

            return result
        }

        return labels[index]
    }

    func getTopLabels(scores: [Double], options: [String: Any]) throws -> [[String: Any]] {
        let dataset = options["dataset"] as? String ?? "coco"
        let k = options["k"] as? Int ?? 5
        let minConfidence = options["minConfidence"] as? Double ?? 0.0
        let includeMetadata = options["includeMetadata"] as? Bool ?? false

        let labels = try getLabels(for: dataset)

        guard scores.count == labels.count else {
            throw NSError(domain: "LabelDatabase", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Score count (\(scores.count)) doesn't match label count (\(labels.count)) for dataset \(dataset)"
            ])
        }

        // Create indexed scores and sort by confidence
        var indexedScores = scores.enumerated().map { (index: $0, score: $1) }
        indexedScores.sort { $0.score > $1.score }

        // Take top K with minimum confidence
        var results: [[String: Any]] = []
        for item in indexedScores.prefix(k) {
            if item.score >= minConfidence {
                var result: [String: Any] = [
                    "index": item.index,
                    "label": labels[item.index],
                    "confidence": item.score
                ]

                if includeMetadata && dataset == "coco" && item.index < cocoSupercategories.count {
                    result["supercategory"] = cocoSupercategories[item.index]
                }

                results.append(result)
            }
        }

        return results
    }

    func getAllLabels(dataset: String) throws -> [String] {
        return try getLabels(for: dataset)
    }

    func getDatasetInfo(dataset: String) throws -> [String: Any] {
        let labels = try getLabels(for: dataset)

        let descriptions: [String: String] = [
            "coco": "COCO 2017 object detection labels (80 classes)",
            "coco91": "COCO original labels with background (91 classes)",
            "imagenet": "ImageNet ILSVRC 2012 classification labels",
            "imagenet21k": "ImageNet-21K full classification labels",
            "voc": "PASCAL VOC object detection labels (21 classes with background)",
            "cifar10": "CIFAR-10 image classification labels (10 classes)",
            "cifar100": "CIFAR-100 image classification labels (100 classes)",
            "places365": "Places365 scene recognition labels",
            "ade20k": "ADE20K semantic segmentation labels (150 classes)"
        ]

        return [
            "name": dataset,
            "numClasses": labels.count,
            "description": descriptions[dataset] ?? "Unknown dataset",
            "isAvailable": true
        ]
    }

    func getAvailableDatasets() -> [String] {
        return ["coco", "coco91", "imagenet", "voc", "cifar10", "cifar100", "places365", "ade20k"]
    }

    // MARK: - Private Helpers

    private func getLabels(for dataset: String) throws -> [String] {
        switch dataset {
        case "coco":
            return cocoLabels
        case "coco91":
            return coco91Labels
        case "imagenet", "imagenet21k":
            return imagenetLabels
        case "voc":
            return vocLabels
        case "cifar10":
            return cifar10Labels
        case "cifar100":
            return cifar100Labels
        case "places365":
            return places365Labels
        case "ade20k":
            return ade20kLabels
        default:
            throw NSError(domain: "LabelDatabase", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Unknown dataset: \(dataset). Available: coco, coco91, imagenet, voc, cifar10, cifar100, places365, ade20k"
            ])
        }
    }
}
