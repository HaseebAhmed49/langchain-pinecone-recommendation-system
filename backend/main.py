from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec  # Use the new Pinecone class
from langchain_openai import OpenAIEmbeddings  # Updated import
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
# Initialize FastAPI
app = FastAPI()

# Pinecone and OpenAI setup
pc = Pinecone(api_key=PINECONE_API_KEY)  # Use the new Pinecone class
index_name = "recommendation-system-demo"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create or connect to a Pinecone index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # 1536 is the dimension of OpenAI embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

# Connect to the Pinecone index
pinecone_index = pc.Index(index_name)  # Get the index object

# Add example data to Pinecone (run once to populate the index)
def add_example_data():
    # Sample item descriptions
    item_descriptions = [
        # Electronics
        "A sleek ultrabook with a 13-inch display, perfect for productivity and portability.",
        "A high-end gaming PC with RGB lighting and liquid cooling.",
        "A wireless earbud with active noise cancellation and 24-hour battery life.",
        "A 4K smart TV with HDR support and built-in streaming apps.",
        "A compact drone with 4K camera and GPS navigation.",
        "A portable projector with 1080p resolution and built-in speakers.",
        "A mechanical keyboard with customizable RGB lighting and tactile switches.",
        "A wireless charging pad compatible with all Qi-enabled devices.",
        "A rugged smartphone with water resistance and long battery life.",
        "A high-fidelity soundbar with Dolby Atmos support.",
        "A foldable tablet with a high-resolution display and stylus support.",
        "A smart thermostat with voice control and energy-saving features.",
        "A wireless security camera with motion detection and night vision.",
        "A portable SSD with 1TB storage and USB-C connectivity.",
        "A gaming mouse with adjustable DPI and ergonomic design.",
        "A VR headset with 6DOF tracking and immersive visuals.",
        "A compact printer with wireless connectivity and high-quality prints.",
        "A smart speaker with voice assistant and multi-room audio support.",
        "A high-speed router with Wi-Fi 6 and mesh networking.",
        "A digital camera with a large sensor and 4K video recording.",

        # Clothing
        "A lightweight down jacket with a hood and water-resistant finish.",
        "A pair of slim-fit jeans with stretch fabric for added comfort.",
        "A cashmere sweater with a classic crew neck design.",
        "A waterproof hiking boot with ankle support and durable soles.",
        "A breathable running shoe with cushioned soles and reflective details.",
        "A tailored blazer with a slim fit and premium wool fabric.",
        "A cozy fleece jacket with a full-zip front and kangaroo pocket.",
        "A cotton t-shirt with a graphic print and relaxed fit.",
        "A silk scarf with a floral pattern and hand-rolled edges.",
        "A leather belt with a polished buckle and adjustable fit.",
        "A pair of wool socks with moisture-wicking technology.",
        "A reversible puffer vest with a lightweight design.",
        "A linen shirt with a button-down collar and relaxed fit.",
        "A pair of yoga pants with high-waist and four-way stretch.",
        "A wool overcoat with a double-breasted design and notch lapel.",
        "A pair of leather gloves with touchscreen compatibility.",
        "A denim jacket with a sherpa-lined collar and button-front closure.",
        "A pair of chino pants with a slim fit and wrinkle-resistant fabric.",
        "A thermal base layer set for cold weather activities.",
        "A pair of running shorts with built-in compression liners.",

        # Home Appliances
        "A smart refrigerator with a touchscreen and Wi-Fi connectivity.",
        "A high-speed blender with multiple preset programs.",
        "A robot vacuum with mapping technology and self-emptying base.",
        "A cordless stick vacuum with powerful suction and lightweight design.",
        "A programmable coffee maker with a built-in grinder.",
        "A countertop ice maker with a compact design and fast production.",
        "A portable air conditioner with a dehumidifier function.",
        "A smart air purifier with HEPA filtration and app control.",
        "A convection microwave oven with sensor cooking technology.",
        "A stand mixer with a 7-quart bowl and multiple attachments.",
        "A slow cooker with a programmable timer and keep-warm function.",
        "A toaster oven with a built-in air fryer and rotisserie.",
        "A compact dishwasher with a quiet operation and energy-saving mode.",
        "A steam mop with a detachable handheld steamer.",
        "A countertop dishwasher with a compact design and quick wash cycle.",
        "A smart thermostat with geofencing and energy-saving features.",
        "A portable washing machine with a compact design and easy setup.",
        "A food processor with multiple blades and a large capacity bowl.",
        "A countertop ice cream maker with a built-in freezer bowl.",
        "A smart ceiling fan with dimmable LED lights and app control.",

        # Outdoor Gear
        "A lightweight tent with a waterproof rainfly and easy setup.",
        "A sleeping bag rated for sub-zero temperatures.",
        "A portable camping stove with a compact design and wind resistance.",
        "A hydration backpack with a 2-liter bladder and multiple compartments.",
        "A pair of trekking poles with adjustable height and shock absorption.",
        "A camping hammock with a mosquito net and rain tarp.",
        "A portable power station with solar charging capability.",
        "A foldable camping chair with a cup holder and side pocket.",
        "A waterproof hiking boot with Vibram soles and ankle support.",
        "A compact binocular with 10x magnification and fog-proof lenses.",
        "A multi-tool with 18 functions and a durable stainless steel body.",
        "A portable grill with a propane tank and temperature control.",
        "A solar-powered lantern with adjustable brightness and USB charging.",
        "A waterproof dry bag with a roll-top closure and multiple sizes.",
        "A portable fire pit with a mesh lid and carry bag.",
        "A camping table with a foldable design and lightweight frame.",
        "A portable water filter with a high flow rate and long lifespan.",
        "A pair of polarized sunglasses with UV protection and anti-glare coating.",
        "A compact fishing rod with a telescopic design and carrying case.",
        "A portable cooler with a 20-liter capacity and ice retention technology.",

        # Fitness and Sports
        "A pair of adjustable dumbbells with a space-saving design.",
        "A yoga mat with a non-slip surface and eco-friendly materials.",
        "A resistance band set with multiple levels of resistance.",
        "A foam roller for muscle recovery and deep tissue massage.",
        "A jump rope with adjustable length and weighted handles.",
        "A pair of boxing gloves with wrist support and breathable mesh.",
        "A fitness tracker with heart rate monitoring and sleep tracking.",
        "A stationary bike with adjustable resistance and a built-in display.",
        "A pull-up bar with multiple grip positions and easy installation.",
        "A kettlebell set with a range of weights and durable coating.",
        "A treadmill with incline adjustment and built-in workout programs.",
        "A rowing machine with a smooth glide and foldable design.",
        "A punching bag with a heavy-duty construction and chain mount.",
        "A pair of running shoes with responsive cushioning and breathable mesh.",
        "A gym bag with multiple compartments and a shoe compartment.",
        "A water bottle with a built-in infuser and leak-proof lid.",
        "A pair of compression sleeves for muscle support and recovery.",
        "A portable exercise bike with a quiet operation and adjustable resistance.",
        "A set of workout bands with door anchors and exercise guide.",
        "A balance board for core strength and stability training.",

        # Beauty and Personal Care
        "A facial cleansing brush with multiple speed settings and waterproof design.",
        "A hair straightener with ceramic plates and adjustable temperature.",
        "A rechargeable electric toothbrush with a 2-minute timer.",
        "A skincare set with a cleanser, toner, and moisturizer.",
        "A hair dryer with ionic technology and multiple heat settings.",
        "A makeup mirror with LED lighting and adjustable brightness.",
        "A set of makeup brushes with synthetic bristles and ergonomic handles.",
        "A lipstick set with long-lasting and moisturizing formulas.",
        "A perfume with floral and woody notes in an elegant bottle.",
        "A men's grooming kit with a trimmer, scissors, and comb.",
        "A body lotion with shea butter and a lightweight formula.",
        "A face mask set with hydrating and detoxifying formulas.",
        "A hair curler with ceramic-coated barrels and heat protection.",
        "A nail polish set with vibrant colors and a quick-dry formula.",
        "A sunscreen with SPF 50 and water-resistant properties.",
        "A beard oil with natural ingredients and a refreshing scent.",
        "A set of sheet masks with collagen and hyaluronic acid.",
        "A hair volumizer with a lightweight and non-greasy formula.",
        "A set of false eyelashes with a natural look and easy application.",
        "A facial roller with a jade stone and cooling effect.",

        # Kitchen and Dining
        "A non-stick frying pan with a durable ceramic coating.",
        "A set of stainless steel knives with a wooden block.",
        "A cast iron skillet with a pre-seasoned finish.",
        "A set of glass food storage containers with airtight lids.",
        "A silicone baking mat with a non-stick surface and heat resistance.",
        "A set of measuring cups and spoons with engraved markings.",
        "A coffee grinder with adjustable grind settings and a large capacity.",
        "A set of reusable silicone straws with a cleaning brush.",
        "A set of bamboo cutting boards with juice grooves.",
        "A set of stainless steel mixing bowls with non-slip bases.",
        "A set of oven mitts with heat-resistant silicone grips.",
        "A set of ceramic mugs with a modern design and comfortable handle.",
        "A set of wine glasses with a thin rim and elegant stem.",
        "A set of stainless steel pots and pans with glass lids.",
        "A set of silicone spatulas with heat-resistant heads.",
        "A set of reusable produce bags with a drawstring closure.",
        "A set of stainless steel tongs with a locking mechanism.",
        "A set of ceramic plates with a minimalist design.",
        "A set of stainless steel utensils with a polished finish.",
        "A set of glass water bottles with a silicone sleeve.",

        # Toys and Games
        "A LEGO set with over 1,000 pieces and a detailed instruction manual.",
        "A board game with strategy and teamwork elements.",
        "A remote-controlled car with a rechargeable battery and high speed.",
        "A puzzle with 1,000 pieces and a vibrant design.",
        "A stuffed animal with soft fur and a huggable size.",
        "A building block set with magnetic pieces and creative designs.",
        "A science kit with experiments for kids and a detailed guide.",
        "A dollhouse with multiple rooms and miniature furniture.",
        "A play kitchen set with realistic sounds and accessories.",
        "A set of action figures with movable joints and accessories.",
        "A toy train set with tracks and a remote control.",
        "A set of art supplies with crayons, markers, and coloring books.",
        "A toy robot with programmable features and interactive modes.",
        "A set of outdoor toys with a frisbee, ball, and jump rope.",
        "A plush toy with a musical feature and soft fabric.",
        "A set of building blocks with interlocking pieces and bright colors.",
        "A toy drone with a camera and easy controls for kids.",
        "A set of dress-up costumes with accessories and props.",
        "A board game with trivia questions and a timer.",
        "A set of toy vehicles with realistic designs and moving parts.",

        # Books and Stationery
        "A hardcover novel with a gripping storyline and beautiful cover.",
        "A set of gel pens with vibrant colors and smooth ink flow.",
        "A planner with a leather cover and monthly layouts.",
        "A notebook with dotted pages and a minimalist design.",
        "A set of highlighters with dual tips and bright colors.",
        "A book on personal development with actionable tips.",
        "A set of sticky notes with assorted colors and sizes.",
        "A set of calligraphy pens with ink cartridges and a guide.",
        "A coloring book with intricate designs and high-quality paper.",
        "A set of markers with fine tips and vibrant colors.",
        "A book on cooking with step-by-step recipes and photos.",
        "A set of pencils with a comfortable grip and durable lead.",
        "A journal with a leather cover and lined pages.",
        "A set of bookmarks with inspirational quotes and tassels.",
        "A book on mindfulness with practical exercises and insights.",
        "A set of watercolor paints with a brush and palette.",
        "A sketchbook with thick paper and a spiral binding.",
        "A set of erasers with fun shapes and a dust-free formula.",
        "A book on productivity with time management strategies.",
        "A set of washi tape with assorted patterns and colors.",

        # Pet Supplies
        "A dog bed with a waterproof liner and removable cover.",
        "A cat tree with multiple levels and scratching posts.",
        "A set of dog toys with squeakers and durable materials.",
        "A pet carrier with a mesh window and padded interior.",
        "A set of cat toys with feathers and bells.",
        "A dog leash with a reflective strip and comfortable handle.",
        "A set of pet bowls with a non-slip base and stainless steel design.",
        "A pet grooming kit with a brush, comb, and nail clippers.",
        "A set of dog treats with natural ingredients and no additives.",
        "A cat litter box with a hood and odor control features.",
        "A dog harness with adjustable straps and a padded chest plate.",
        "A set of pet wipes with a gentle formula for cleaning.",
        "A pet collar with a breakaway buckle and reflective stitching.",
        "A set of catnip toys with a variety of shapes and textures.",
        "A dog crate with a foldable design and removable tray.",
        "A set of pet shampoo with a hypoallergenic formula.",
        "A pet food dispenser with a timer and portion control.",
        "A set of dog training pads with a leak-proof design.",
        "A cat scratching post with a sisal rope and sturdy base.",
        "A pet first aid kit with essential supplies and a guide."
    ]
    # Generate embeddings for the item descriptions
    item_embeddings = embeddings.embed_documents(item_descriptions)
    # Prepare data for upsert
    vectors = [
        {
            "id": f"item_{i}",
            "values": embedding,
            "metadata": {"text": item_descriptions[i]}
        }
        for i, embedding in enumerate(item_embeddings)
    ]
    # Upsert vectors into Pinecone
    pinecone_index.upsert(vectors=vectors)

# Uncomment the line below to add example data to Pinecone (run once)
# add_example_data()

# Pydantic model for request body
class RecommendationRequest(BaseModel):
    item_description: str
    k: int = 5  # Number of recommendations to return

# Recommendation endpoint
@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    try:
        # Generate embedding for the query
        query_embedding = embeddings.embed_query(request.item_description)
        # Query Pinecone index
        query_response = pinecone_index.query(
            vector=query_embedding,
            top_k=request.k,
            include_metadata=True
        )
        # Format results
        recommendations = [{"item": match.metadata["text"]} for match in query_response.matches]
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)