# -*- coding: utf-8 -*-
import os
import json
import torch
import gradio as ui
import soundfile as sf
from transformers import VitsConfig, VitsModel, VitsTokenizer

# Зареждаме нашия пренаписан тренировъчен модул
import train_ms
import logging
import sys

import glob


my_dir_path = os.path.dirname(os.path.abspath(__file__))

def get_latest_g_checkpoint(folder_path=my_dir_path + "/My_Model_Folder"):
    """Автоматично намира най-новия G_*.pth файл в посочената папка"""
    # Търсим всички файлове, съвпадащи с шаблона G_*.pth
    search_pattern = os.path.join(folder_path, "G_*.pth")
    files = glob.glob(search_pattern)
    
    if not files:
        # Ако няма намерени файлове, връщаме базова стойност, за да не е празно
        return os.path.join(folder_path, "G_3047200.pth")
        
    # Подреждаме файловете по време на последна промяна (най-новият последен)
    latest_file = max(files, key=os.path.getmtime)
    
    # Конвертираме наклонените черти в нормален Windows стил за красота
    return os.path.normpath(latest_file)

def start_training_ui(run_name, dataset_path, val_path, model_dir, epochs, batch_size, lr, fp16):
    """Функция, която улавя настройките от интерфейса, записва ги в config.json и стартира train_ms.py"""
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.json")
    if not os.path.exists(config_path):
        return f"❌ Грешка: Не е намерен базов config.json в папка 'configs/'!"
        
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
        
    config_data["data"]["training_files"] = dataset_path
    config_data["data"]["validation_files"] = val_path
    config_data["train"]["epochs"] = int(epochs)
    config_data["train"]["batch_size"] = int(batch_size)
    config_data["train"]["learning_rate"] = float(lr)
    config_data["train"]["fp16_run"] = fp16
    config_data["model_dir"] = model_dir
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
        
    print(f"⚙️ Конфигурацията за [{run_name}] е обновена успешно!")
    
    try:
        import sys
        sys.argv = ["train_ms.py", "-c", config_path, "-m", model_dir]
        ui.Info("🔥 Обучението стартира на твоята RTX 4060! Провери CMD конзолата за прогреса.")
        train_ms.main()
        return f"🎉 Успех! Обучението за '{run_name}' завърши. Моделите са записани в {model_dir}"
    except Exception as e:
        return f"❌ Софтуерен срив по време на обучение: {str(e)}"


# 🧙‍♂️ НАШИЯТ СПАСИТЕЛЕН МАГИЧЕСКИ ПРЕВОДАЧ НА КЛЮЧОВЕ
def translate_vits_to_hf(old_key):
    MAPPING = {
        "enc_p.emb.weight": "text_encoder.embed_tokens.weight",
        "enc_p.proj": "text_encoder.project",
        "enc_q.pre": "posterior_encoder.conv_pre",
        "enc_q.proj": "posterior_encoder.conv_proj",
        "enc_q.enc.cond_layer": "posterior_encoder.wavenet.cond_layer",
        "emb_g": "embed_speaker.weight",
        "dec.conv_pre": "decoder.conv_pre",
        "dec.conv_post": "decoder.conv_post",
        "dec.cond": "decoder.cond"
    }
    if old_key in MAPPING:
        return MAPPING[old_key]
    
    new_key = old_key
    if "enc_p.encoder.attn_layers" in old_key:
        new_key = old_key.replace("enc_p.encoder.attn_layers", "text_encoder.encoder.layers")
        new_key = new_key.replace(".conv_q.", ".attention.q_proj.")
        new_key = new_key.replace(".conv_k.", ".attention.k_proj.")
        new_key = new_key.replace(".conv_v.", ".attention.v_proj.")
        new_key = new_key.replace(".conv_o.", ".attention.out_proj.")
        new_key = new_key.replace(".emb_rel_k", ".attention.emb_rel_k")
        new_key = new_key.replace(".emb_rel_v", ".attention.emb_rel_v")
    elif "enc_p.encoder.ffn_layers" in old_key:
        new_key = old_key.replace("enc_p.encoder.ffn_layers", "text_encoder.encoder.layers")
    elif "enc_p.encoder.norm_layers_1" in old_key:
        new_key = old_key.replace("enc_p.encoder.norm_layers_1", "text_encoder.encoder.layers")
        new_key = new_key.replace(".gamma", ".layer_norm.weight").replace(".beta", ".layer_norm.bias")
    elif "enc_p.encoder.norm_layers_2" in old_key:
        new_key = old_key.replace("enc_p.encoder.norm_layers_2", "text_encoder.encoder.layers")
        new_key = new_key.replace(".gamma", ".final_layer_norm.weight").replace(".beta", ".final_layer_norm.bias")
    elif "enc_q.enc.in_layers" in old_key:
        new_key = old_key.replace("enc_q.enc.in_layers", "posterior_encoder.wavenet.in_layers")
    elif "enc_q.enc.res_skip_layers" in old_key:
        new_key = old_key.replace("enc_q.enc.res_skip_layers", "posterior_encoder.wavenet.res_skip_layers")
    elif "dec.ups" in old_key:
        new_key = old_key.replace("dec.ups", "decoder.upsampler")
    elif "dec.resblocks" in old_key:
        new_key = old_key.replace("dec.resblocks", "decoder.resblocks")
        
    return new_key


def test_voice_ui(model_path, text, config_json_path, vocab_txt_path):
    """Генерира глас в реално време, използвайки резервни части от Facebook, за да няма шум"""
    BASE_MODEL_ID = "facebook/mms-tts-bul"
    
    if not os.path.exists(model_path):
        return None, "❌ Грешка: Посоченият .pth модел не съществува!"
    if not os.path.exists(config_json_path) or not os.path.exists(vocab_txt_path):
        return None, "❌ Грешка: Липсва config.json или vocab.txt за конфигуриране!"
        
    try:
        ui.Info("🎙️ Зареждам кристално чистия филтър за тегла...")
        
        # 1. Зареждаме базата за резервни части
        base_hf_model = VitsModel.from_pretrained(BASE_MODEL_ID)
        base_state_dict = base_hf_model.state_dict()
        
        # 2. Зареждаме твоя трениран файл
        checkpoint = torch.load(model_path, map_location="cpu")
        trained_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        
        # 3. Превеждаме твоя файл
        translated_trained_dict = {}
        for old_k, tensor in trained_dict.items():
            new_k = translate_vits_to_hf(old_k)
            if any(proj in new_k for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                if tensor.dim() == 3 and tensor.shape[-1] == 1:
                    tensor = tensor.squeeze(-1)
            translated_trained_dict[new_k] = tensor

        # 4. Сглобяваме без празни дупки
        final_state_dict = {}
        for hf_key, hf_tensor in base_state_dict.items():
            if hf_key in translated_trained_dict:
                final_state_dict[hf_key] = translated_trained_dict[hf_key].clone()
            else:
                final_state_dict[hf_key] = hf_tensor.clone()

        # 5. Инициализираме модела с твоята конфигурация
#       config = VitsConfig.from_pretrained(config_json_path)

#       print("config_json_path: ", config)

        # 5. Инициализираме модела с твоята конфигурация по сигурния начин
        with open(config_json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = VitsConfig(**config_dict)

        config.vocab_size = len([line for line in open(vocab_txt_path, encoding="utf-8").readlines()])

        config.num_speakers = 1
        config.speaker_embedding_size = 0
        
        model = VitsModel(config)

        model.load_state_dict(final_state_dict, strict=True)
        model.eval()
       
        # 6. Токенизация и изговаряне
        tokenizer = VitsTokenizer(temp_name=None, vocab_file=my_dir_path + "/configs/vocab.json", language="bul", phonemize=False, is_uroman=False, pad_token="_")

        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            output = model(**inputs)
        
        output_wav = "webui_test_output.wav"
        sf.write(output_wav, output.waveform[0].numpy(), 16000)
        
        return output_wav, "🎵 Браво! Твоят глас проговори кристално ясно директно в уеб плеъра!"
    except Exception as e:
        return None, f"❌ Грешка при генериране: {str(e)}"


def export_hf_model_ui(model_path, config_json_path, vocab_txt_path, output_dir):
    """Специален нов бутон, който пакетира модела за секунда в чист Transformers формат"""
    if not os.path.exists(model_path):
        return "❌ Грешка: Невалиден път до G_*.pth модела!"
    try:
        ui.Info("🔮 Пакетирам модела за Hugging Face...")
        base_hf_model = VitsModel.from_pretrained("facebook/mms-tts-bul")
        base_state_dict = base_hf_model.state_dict()
        
        checkpoint = torch.load(model_path, map_location="cpu")
        trained_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        
        translated_trained_dict = {}
        for old_k, tensor in trained_dict.items():
            new_k = translate_vits_to_hf(old_k)
            if any(proj in new_k for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                if tensor.dim() == 3 and tensor.shape[-1] == 1:
                    tensor = tensor.squeeze(-1)
            translated_trained_dict[new_k] = tensor

        final_state_dict = {}
        for hf_key, hf_tensor in base_state_dict.items():
            if hf_key in translated_trained_dict:
                final_state_dict[hf_key] = translated_trained_dict[hf_key].clone()
            else:
                final_state_dict[hf_key] = hf_tensor.clone()

        config = VitsConfig.from_pretrained(config_json_path)

        print("config_json_path: ", config_json_path)

#       with open(config_json_path, "r", encoding="utf-8") as f:
#           config_dict = json.load(f)
#       config = VitsConfig(**config_dict)

        config.vocab_size = len([line for line in open(vocab_txt_path, encoding="utf-8").readlines()])

        config.num_speakers = 1
        config.speaker_embedding_size = 0
        
        final_model = VitsModel(config)
        final_model.load_state_dict(final_state_dict, strict=True)
        
        os.makedirs(output_dir, exist_ok=True)
        final_model.save_pretrained(output_dir)
        
        tokenizer = VitsTokenizer(temp_name=None, vocab_file=my_dir_path + "/configs/vocab.json", language="bul", phonemize=False, is_uroman=False, pad_token="_")
        tokenizer.save_pretrained(output_dir)
        
        return f"🏆 УСПЕХ! Чистият Hugging Face модел е записан в папка: '{output_dir}'. Вече можеш да го копираш или качиш в интернет!"
    except Exception as e:
        return f"❌ Грешка при експортиране: {str(e)}"


# ==========================================
# СГЛОБЯВАНЕ НА GRADIO ИНТЕРФЕЙСА (WEBUI)
# ==========================================
with ui.Blocks(title="VITS MMS Bulgarian Voice Clone WebUI") as demo:
    ui.Markdown("# 🎙️ VITS Bulgarian Voice Clone WebUI (Doc Brown Edition ⚡)")
    ui.Markdown("Добре дошъл в твоя обновен команден пулт. Сега всичко работи автоматично и без пращене.")
                
    with ui.Tabs():
        # ТАБ 1: ОБУЧЕНИЕ
        with ui.TabItem("🏋️‍♂️ Обучение (Train)"):
            ui.Markdown("### Настройки на тренировъчния процес")
            with ui.Row():
                run_name = ui.Textbox(label="Име на тренировката", value="My_Bulgarian_Voice_v1")
                model_dir = ui.Textbox(label="Папка за запис на моделите (D_*.pth и G_*.pth)", value=my_dir_path + "/My_Model_Folder")
            with ui.Row():
                dataset_path = ui.Textbox(label="Път до тренировъчния датасет (.txt)", value=my_dir_path + "/configs/train.txt")
                val_path = ui.Textbox(label="Път до валидационния датасет (.txt)", value=my_dir_path + "/configs/dev.txt")
            with ui.Row():
                epochs = ui.Slider(minimum=1, maximum=10000, value=100, step=1, label="Брой Епохи (Epochs)")
                batch_size = ui.Slider(minimum=1, maximum=32, value=4, step=1, label="Batch Size (Препоръчително: 2-4 за 4060)")
                lr = ui.Textbox(label="Скорост на обучение (Learning Rate)", value="1e-5")
                fp16 = ui.Checkbox(label="Активирай FP16 (По-бързо смятане и по-малко памет)", value=True)
                
            train_btn = ui.Button("🔥 СТАРТИРАЙ ОБУЧЕНИЕТО", variant="primary")
            train_output = ui.Textbox(label="Статус на тренировката")
            
            train_btn.click(
                start_training_ui, 
                inputs=[run_name, dataset_path, val_path, model_dir, epochs, batch_size, lr, fp16], 
                outputs=train_output
            )
           
        # ТАБ 2: ТЕСТВАНЕ 
        with ui.TabItem("🎙️ Тестване на модела (Inference)"):
            ui.Markdown("### Тествай твоя дообучен модел директно тук")
            with ui.Row():
                model_path = ui.Textbox(label="Път до твоя финална G_*.pth точка", value=get_latest_g_checkpoint()) # my_dir_path + "/My_Model_Folder/G_3081000.pth")
            with ui.Row():
                config_json_path = ui.Textbox(label="Път до твоя config.json", value=my_dir_path + "/configs/config.json")
                vocab_txt_path = ui.Textbox(label="Път до твоя vocab.txt", value=my_dir_path + "/configs/vocab.txt")
            
            test_text = ui.Textbox(
                label="Текст за изговаряне", 
                value="Здравей! Това е гласов тест на новия ми обновен уеб интерфейс. Вече всичко се чува прекрасно.", 
                lines=3
            )
            
            test_btn = ui.Button("🎵 ГЕНЕРИРАЙ КРИСТАЛЕН ГЛАС", variant="secondary")
            
            with ui.Row():
                audio_player = ui.Audio(label="Генерирано аудио", type="filepath")
                test_status = ui.Textbox(label="Статус на теста")
                
            test_btn.click(
                test_voice_ui,
                inputs=[model_path, test_text, config_json_path, vocab_txt_path],
                outputs=[audio_player, test_status]
            )

        # ТАБ 3: НОВИЯТ ЕКСПОРТЕН ТАБ За Разкрасяване!
        with ui.TabItem("🔮 HF Експорт (Save & Export)"):
            ui.Markdown("### Пакетирай твоя трениран файл в готов модел за интернет")
            with ui.Row():
                exp_model_path = ui.Textbox(label="Избери тренирания G_*.pth файл за експорт", value=get_latest_g_checkpoint()) # my_dir_path + "/My_Model_Folder/G_3081000.pth")
                exp_output_dir = ui.Textbox(label="Папка, където да се запише финалния модел", value=my_dir_path + "/my-final-huggingface-model")
            with ui.Row():
                exp_config_path = ui.Textbox(label="Път до твоя config.json", value=my_dir_path + "/configs/config.json")
                exp_vocab_path = ui.Textbox(label="Път до твоя vocab.txt", value=my_dir_path + "/configs/vocab.txt")
                
            export_btn = ui.Button("🏆 КОНВЕРТИРАЙ И ЗАПИШИ ЗА HUGGING FACE", variant="primary")
            export_status = ui.Textbox(label="Резултат от експорта")
            
            export_btn.click(
                export_hf_model_ui,
                inputs=[exp_model_path, exp_config_path, exp_vocab_path, exp_output_dir],
                outputs=export_status
            )

if __name__ == "__main__":
    demo.launch(inbrowser=True)