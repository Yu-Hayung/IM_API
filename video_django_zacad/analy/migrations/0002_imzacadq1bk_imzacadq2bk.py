# Generated by Django 3.2.4 on 2021-08-30 05:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analy', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ImZacadQ1Bk',
            fields=[
                ('zq_pk', models.AutoField(db_column='ZQ_PK', primary_key=True, serialize=False)),
                ('zq_code', models.IntegerField(blank=True, db_column='ZQ_CODE', null=True)),
                ('zq_q1', models.CharField(blank=True, db_column='ZQ_Q1', max_length=200, null=True)),
                ('zq_a1', models.CharField(blank=True, db_column='ZQ_A1', max_length=2000, null=True)),
            ],
            options={
                'db_table': 'IM_ZACAD_Q1_BK',
            },
        ),
        migrations.CreateModel(
            name='ImZacadQ2Bk',
            fields=[
                ('zq_pk', models.AutoField(db_column='ZQ_PK', primary_key=True, serialize=False)),
                ('zq_q2', models.CharField(blank=True, db_column='ZQ_Q2', max_length=200, null=True)),
            ],
            options={
                'db_table': 'IM_ZACAD_Q2_BK',
            },
        ),
    ]