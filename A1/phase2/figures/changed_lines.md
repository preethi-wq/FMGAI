# Changes to train_instruct_pix2pix.py (diffusers v0.41.0 example)

14 change blocks; 13 lines marked `# MODIFIED`, 54 lines marked `# ADDED`.

## Change 1 (insert)

**Original**, (insert after line 17):

```python
    (nothing)
```

**Modified**, lines 18-18:

```python
  18  EXPONENT_X = 1.60  # ADDED: declared exponent x in (1.5, 3.0); Run B fine-tunes with {beta_t ** EXPONENT_X}
```

## Change 2 (insert)

**Original**, (insert after line 19):

```python
    (nothing)
```

**Modified**, lines 21-21:

```python
  21  import json  # ADDED
```

## Change 3 (insert)

**Original**, (insert after line 54):

```python
    (nothing)
```

**Modified**, lines 57-57:

```python
  57  import sr_common  # ADDED: shared data, schedule, sampling and metric helpers for this assignment
```

## Change 4 (insert)

**Original**, (insert after line 398):

```python
    (nothing)
```

**Modified**, lines 402-411:

```python
 402      parser.add_argument("--data_dir", type=str, default=None, help="Folder written by prepare_data.py.")  # ADDED
 403      parser.add_argument(  # ADDED
 404          "--noise_schedule",  # ADDED
 405          type=str,  # ADDED
 406          default="original",  # ADDED
 407          choices=["original", "power"],  # ADDED
 408          help="original = {beta_t} (Run A); power = {beta_t ** EXPONENT_X} (Run B).",  # ADDED
 409      )  # ADDED
 410      parser.add_argument("--num_sample_checkpoints", type=int, default=5, help="Equally spaced sample epochs.")  # ADDED
 411      parser.add_argument("--num_val_metric_images", type=int, default=32, help="Val images scored per checkpoint.")  # ADDED
```

## Change 5 (replace)

**Original**, lines 406-406:

```python
 406      if args.dataset_name is None and args.train_data_dir is None:
```

**Modified**, lines 419-419:

```python
 419      if args.dataset_name is None and args.train_data_dir is None and args.data_dir is None:  # MODIFIED
```

## Change 6 (insert)

**Original**, (insert after line 491):

```python
    (nothing)
```

**Modified**, lines 505-515:

```python
 505      original_betas = noise_scheduler.betas.clone()  # ADDED: native theta_t = beta_t of the pretrained model
 506      sr_common.check_theta_in_unit_interval(original_betas)  # ADDED
 507      if args.noise_schedule == "power":  # ADDED
 508          noise_scheduler = DDPMScheduler.from_config(  # ADDED
 509              noise_scheduler.config, trained_betas=sr_common.power_betas(original_betas, EXPONENT_X).tolist()  # ADDED
 510          )  # ADDED
 511      run_betas = noise_scheduler.betas.clone()  # ADDED
 512      sr_common.verify_run_schedule(run_betas, original_betas, args.noise_schedule, EXPONENT_X)  # ADDED
 513      logger.info(f"Noise schedule '{args.noise_schedule}': {sr_common.schedule_summary(run_betas)}")  # ADDED
 514      if accelerator.is_main_process:  # ADDED
 515          sr_common.save_schedule(args.output_dir, run_betas, original_betas, args.noise_schedule, EXPONENT_X)  # ADDED
```

## Change 7 (replace)

**Original**, lines 510-521:

```python
 510      logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
 511      in_channels = 8
 512      out_channels = unet.conv_in.out_channels
 513      unet.register_to_config(in_channels=in_channels)
 514  
 515      with torch.no_grad():
 516          new_conv_in = nn.Conv2d(
 517              in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
 518          )
 519          new_conv_in.weight.zero_()
 520          new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
 521          unet.conv_in = new_conv_in
```

**Modified**, lines 534-545:

```python
 534      if unet.config.in_channels == 4:  # ADDED: the InstructPix2Pix checkpoint already has 8 input channels
 535          logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")  # MODIFIED
 536          in_channels = 8  # MODIFIED
 537          out_channels = unet.conv_in.out_channels  # MODIFIED
 538          unet.register_to_config(in_channels=in_channels)  # MODIFIED
 539          with torch.no_grad():  # MODIFIED
 540              new_conv_in = nn.Conv2d(  # MODIFIED
 541                  in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding  # MODIFIED
 542              )  # MODIFIED
 543              new_conv_in.weight.zero_()  # MODIFIED
 544              new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)  # MODIFIED
 545              unet.conv_in = new_conv_in  # MODIFIED
```

## Change 8 (replace)

**Original**, lines 624-624:

```python
 624      if args.dataset_name is not None:
```

**Modified**, lines 648-650:

```python
 648      if args.data_dir is not None:  # ADDED
 649          dataset = sr_common.load_sr_dataset(args.data_dir)  # ADDED: paired (LR input, HR target) training split
 650      elif args.dataset_name is not None:  # MODIFIED
```

## Change 9 (insert)

**Original**, (insert after line 816):

```python
    (nothing)
```

**Modified**, lines 843-848:

```python
 843      val_rows = sr_common.read_split(args.data_dir, "val") if args.data_dir is not None else []  # ADDED
 844      sample_epochs = sr_common.equally_spaced_epochs(args.num_train_epochs, args.num_sample_checkpoints)  # ADDED
 845      run_label = "Run B" if args.noise_schedule == "power" else "Run A"  # ADDED
 846      logger.info(f"  {run_label}: samples and validation metrics at epochs {sample_epochs}")  # ADDED
 847      metrics_cache = {}  # ADDED
 848      schedule_checked = False  # ADDED
```

## Change 10 (insert)

**Original**, (insert after line 851):

```python
    (nothing)
```

**Modified**, lines 884-884:

```python
 884          epoch_loss_sum, epoch_batches = 0.0, 0  # ADDED: mean train loss per epoch for train_log.csv
```

## Change 11 (insert)

**Original**, (insert after line 875):

```python
    (nothing)
```

**Modified**, lines 909-917:

```python
 909                  if not schedule_checked:  # ADDED: one-time proof that the forward process uses this run's schedule
 910                      check = sr_common.verify_add_noise(  # ADDED
 911                          noisy_latents, latents, noise, timesteps, original_betas, args.noise_schedule, EXPONENT_X  # ADDED
 912                      )  # ADDED
 913                      logger.info(f"Forward-process schedule check passed: {check}")  # ADDED
 914                      if accelerator.is_main_process:  # ADDED
 915                          with open(os.path.join(args.output_dir, "schedule_check.json"), "w") as f:  # ADDED
 916                              json.dump(check, f, indent=1)  # ADDED
 917                      schedule_checked = True  # ADDED
```

## Change 12 (insert)

**Original**, (insert after line 922):

```python
    (nothing)
```

**Modified**, lines 965-966:

```python
 965                  epoch_loss_sum += avg_loss.item()  # ADDED
 966                  epoch_batches += 1  # ADDED
```

## Change 13 (insert)

**Original**, (insert after line 971):

```python
    (nothing)
```

**Modified**, lines 1016-1021:

```python
1016          if accelerator.is_main_process and epoch_batches > 0:  # ADDED: CSV log, validation, checkpoint samples
1017              sr_common.end_of_epoch(  # ADDED
1018                  epoch + 1, epoch_loss_sum / epoch_batches, args, run_label, val_rows, sample_epochs,  # ADDED
1019                  unwrap_model(unet), vae, text_encoder, tokenizer, noise_scheduler,  # ADDED
1020                  accelerator.device, weight_dtype, metrics_cache,  # ADDED
1021              )  # ADDED
```

## Change 14 (insert)

**Original**, (insert after line 1018):

```python
    (nothing)
```

**Modified**, lines 1069-1071:

```python
1069              scheduler=sr_common.make_ddim(noise_scheduler.config, run_betas),  # ADDED: save this run's schedule
1070              safety_checker=None,  # ADDED: the NSFW checker is never used for SR; skipping it avoids a 1.2 GB load
1071              requires_safety_checker=False,  # ADDED
```
