# Generated by Django 5.0.4 on 2024-05-18 02:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("s2s", "0020_merge_20240518_0220"),
    ]

    operations = [
        migrations.AddField(
            model_name="event",
            name="price",
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name="profile",
            name="last_email_sent",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="userevents",
            name="user_rating",
            field=models.CharField(
                choices=[
                    ("Did not attend", "Did not attend"),
                    ("Not Rated", "Not Rated"),
                    ("1", "1"),
                    ("2", "2"),
                    ("3", "3"),
                    ("4", "4"),
                    ("5", "5"),
                ],
                default="Not Rated",
                max_length=15,
            ),
        ),
    ]
